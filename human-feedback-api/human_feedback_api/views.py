import random
from collections import namedtuple
from datetime import timedelta, datetime

from django import template
from django.db.models import Q
from django.shortcuts import get_object_or_404
from django.shortcuts import render
from django.utils import timezone

from human_feedback_api.models import Comparison
from human_feedback_api.models import SortTree
from human_feedback_api.models import Clip

import human_feedback_api.redblack_tree as redblack

register = template.Library()

ExperimentResource = namedtuple("ExperimentResource", ['name', 'num_responses', 'started_at', 'pretty_time_elapsed'])

def _pretty_time_elapsed(start, end):
    total_seconds = (end - start).total_seconds()
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return ("{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))

def _build_experiment_resource(experiment_name):
    comparisons = Comparison.objects.filter(experiment_name=experiment_name, responded_at__isnull=False)
    try:
        started_at = comparisons.order_by('-created_at').first().created_at
        pretty_time_elapsed = _pretty_time_elapsed(started_at, timezone.now())
    except AttributeError:
        started_at = None
        pretty_time_elapsed = None
    return ExperimentResource(
        name=experiment_name,
        num_responses=comparisons.count(),
        started_at=started_at,
        pretty_time_elapsed=pretty_time_elapsed
    )

def _all_comparisons(experiment_name, use_locking=True):
    not_responded = Q(responded_at__isnull=True)

    if use_locking:
        cutoff_time = timezone.now() - timedelta(minutes=2)
        not_in_progress = Q(shown_to_tasker_at__isnull=True) | Q(shown_to_tasker_at__lte=cutoff_time)
        finished_uploading_media = Q(created_at__lte=datetime.now() - timedelta(seconds=2))  # Give time for upload
        ready = not_responded & not_in_progress & finished_uploading_media
    else:
        ready = not_responded

    # Sort by priority, then put newest labels first
    return Comparison.objects.filter(ready, experiment_name=experiment_name).order_by('-priority', '-created_at')

def index(request):
    return render(request, 'index.html', context={
        'old_experiment_names': [
            exp for exp in
            Comparison.objects.order_by().values_list('experiment_name', flat=True).distinct()],
        'new_experiment_names': [
            exp for exp in
            SortTree.objects.filter(parent=None).order_by().values_list('experiment_name', flat=True).distinct()]
    })

def list_comparisons(request, experiment_name):
    comparisons = Comparison.objects.filter(experiment_name=experiment_name).order_by('responded_at', '-priority')
    return render(request, 'list.html', context=dict(comparisons=comparisons, experiment_name=experiment_name))

def display_comparison(comparison):
    """Mark comparison as having been displayed"""
    comparison.shown_to_tasker_at = timezone.now()
    comparison.save()

def ajax_response(request, experiment_name):
    """Update a comparison with a response"""

    POST = request.POST
    comparison_id = POST.get("comparison_id")
    debug = True

    comparison = Comparison.objects.get(pk=comparison_id)

    # Update the values
    comparison.response = POST.get("response")
    comparison.responded_at = timezone.now()

    if debug:
        print("Answered comparison {} with {}".format(comparison_id, comparison.response))

    comparison.full_clean()  # Validation
    comparison.save()

    # If this comparison belongs to a sorting tree, run the tree logic...
    _sorting_logic(experiment_name)

    comparisons = list(_all_comparisons(experiment_name)[:1])
    for comparison in comparisons:
        display_comparison(comparison)
    if debug:
        print("{}".format([x.id for x in comparisons]))
        if comparison:
            print("Requested {}".format(comparison.id))
    return render(request, 'ajax_response.html', context={
        'comparisons': comparisons,
        'experiment': _build_experiment_resource(experiment_name)
    })

def show_comparison(request, comparison_id):
    comparison = get_object_or_404(Comparison, pk=comparison_id)
    return render(request, 'show_comparison.html', context={"comparison": comparison})

def respond(request, experiment_name):
    comparisons = list(_all_comparisons(experiment_name)[:3])
    for comparison in comparisons:
        display_comparison(comparison)

    return render(request, 'responses.html', context={
        'comparisons': comparisons,
        'experiment': _build_experiment_resource(experiment_name)
    })

def all_clips(request, environment_id):
    return render(request, 'all_clips.html', context={"clips": Clip.objects.filter(environment_id=environment_id)})

# New interface
def _handle_node_pending_clips(node, experiment_name):
    any_comparisons = len(Comparison.objects.filter(tree_node=node, response=None)) > 0
    for clip1 in node.pending_clips.all():
        try:
            comp = Comparison.objects.get(tree_node=node, media_url_1=clip1.media_url)
            if comp.response:
                if comp.response in ["left", "right"]:
                    # Okay, this is a little bit weird. Sorry for the awkwardness.
                    # When the user responds "left" they're saying that the left clip is BETTER
                    # In the redblack tree, left-nodes are worse, and right-nodes are better.
                    # Since clip1 is the left clip, we want to move it left in the tree if the user says "right".
                    is_worse = (comp.response == "right")
                    redblack.move_clip_down(node, clip1, is_worse)
                    return clip1
                else:  # Assume tie
                    node.bound_clips.add(clip1)
                    return clip1
            # else:
            #    still waiting on response for this comparison
        except Comparison.DoesNotExist:
            if not any_comparisons:
                clip2 = random.choice(node.bound_clips.all())
                print("Let's make a comparison between", clip1, clip1.clip_tracking_id, "and", clip2, clip2.clip_tracking_id)

                comparison = Comparison(
                    experiment_name=experiment_name,
                    media_url_1=clip1.media_url,
                    media_url_2=clip2.media_url,
                    response_kind='left_or_right',
                    priority=1.,
                    tree_node=node,
                )
                comparison.full_clean()
                comparison.save()
                any_comparisons = True
                return None
    return None

def _sorting_logic(experiment_name):
    run_logic = True
    while run_logic:
        run_logic = False
        # Look to generate comparisons from the tree
        active_tree_nodes = SortTree.objects.filter(experiment_name=experiment_name).exclude(pending_clips=None)
        for node in active_tree_nodes:
            clip_to_remove = _handle_node_pending_clips(node, experiment_name)
            if clip_to_remove:
                node.pending_clips.remove(clip_to_remove)
                # If a clip was sorted, we want to stop looping and run the logic over from the start
                # This helps generate comparisons as soon as they're available and avoid concurrent sort bugs
                run_logic = True
                break

def compare(request, experiment_name):
    _sorting_logic(experiment_name)

    return respond(request, experiment_name)

def _get_visnodes(node, depth, tree_position, what_kind_of_child_i_am):
    max_depth = depth
    results = [{
        'id': 'visnode%s' % node.id,
        'bound_clips': [clip.media_url for clip in node.bound_clips.all()],
        'show_clip': 0,
        'tree_position': tree_position,  # If the root pos=1, this ranges (0, 2)
        'depth': depth,
        'color': '#A00' if node.is_red else 'black',
        'text_color': 'white',
        'what_kind_of_child_i_am': what_kind_of_child_i_am,
        'num_bound_clips': len(node.bound_clips.all()),
        'num_pending_clips': len(node.pending_clips.all()),
    }]
    if node.right:
        right_position = tree_position + (0.5**(depth + 1))
        sub_visnodes, max_subdepth = _get_visnodes(node.right, depth + 1, right_position, 'right')
        results += sub_visnodes
        max_depth = max(max_depth, max_subdepth)
    if node.left:
        left_position = tree_position - (0.5**(depth + 1))
        sub_visnodes, max_subdepth = _get_visnodes(node.left, depth + 1, left_position, 'left')
        results += sub_visnodes
        max_depth = max(max_depth, max_subdepth)
    return results, max_depth

def _set_visnode_position_data(visnodes, max_depth, clip_width):
    clip_plus_frame_width = clip_width + 20
    clip_plus_frame_height = clip_width + 116
    largest_row = 2 ** max_depth
    total_width = clip_plus_frame_width * largest_row
    total_height = (max_depth + 1) * clip_plus_frame_height
    for vn in visnodes:
        vn['top_edge'] = vn['depth'] * clip_plus_frame_height
        vn['left_edge'] = total_width * (vn['tree_position'] / 2)
        vn['x_position'] = vn['left_edge'] + (clip_plus_frame_width / 2)
        vn['y_position'] = vn['top_edge'] + (clip_plus_frame_height / 2)
        shift = 0.5 ** vn['depth']  # How much we have to move over in tree_position to get to the parent
        if vn['what_kind_of_child_i_am'] == "left":
            vn['parent_x_pos'] = vn['x_position'] + (total_width * (shift / 2))
            vn['parent_y_pos'] = vn['y_position'] - clip_plus_frame_height
        elif vn['what_kind_of_child_i_am'] == "right":
            vn['parent_x_pos'] = vn['x_position'] - (total_width * (shift / 2))
            vn['parent_y_pos'] = vn['y_position'] - clip_plus_frame_height
        else:
            vn['parent_x_pos'] = vn['x_position']
            vn['parent_y_pos'] = vn['y_position']
    return total_width, total_height

def tree(request, experiment_name):
    root = SortTree.objects.get(experiment_name=experiment_name, parent=None)
    visnodes, max_depth = _get_visnodes(root, depth=0, tree_position=1, what_kind_of_child_i_am=None)
    dim = _set_visnode_position_data(visnodes, max_depth, 84)
    return render(request, 'tree.html', context={"tree": visnodes, "total": {"width": dim[0], "height": dim[1]}})
