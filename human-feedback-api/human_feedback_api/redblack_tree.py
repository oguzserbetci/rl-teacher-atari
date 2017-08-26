from human_feedback_api.models import SortTree

def _root(tree):
    while tree.parent:
        tree = tree.parent
    return tree

def _left_rotate(x):
    """ Does a left tree-rotation around node x. """
    # See https://en.wikipedia.org/wiki/Tree_rotation
    # Logic drawn from https://github.com/headius/redblack
    y = x.right
    x.set_right(y.left)
    if x.parent:
        if x == x.parent.left:
            x.parent.set_left(y)
        else:
            x.parent.set_right(y)
    else:  # y becomes new root
        y.parent = None
        y.save()
    y.set_left(x)

def _right_rotate(x):
    """ Does a right tree-rotation around node x. """
    # See https://en.wikipedia.org/wiki/Tree_rotation
    # Logic drawn from https://github.com/headius/redblack
    y = x.left
    x.set_left(y.right)
    if x.parent:
        if x == x.parent.left:
            x.parent.set_left(y)
        else:
            x.parent.set_right(y)
    else:  # y becomes new root
        y.parent = None
        y.save()
    y.set_right(x)

def _rebalance_tree(new_node):
    """ Rebalances a redblack tree after adding a node. """
    # Logic drawn from https://github.com/headius/redblack
    node = new_node
    while node.parent and node.parent.is_red:
        # Note: Since the root to the tree is always black, node must have a grandparent.
        if node.parent == node.parent.parent.left:  # node's parent is a left-child
            uncle = node.parent.parent.right
            if uncle and uncle.is_red:
                node.parent.make_black()
                uncle.make_black()
                node.parent.parent.make_red()
                node = node.parent.parent
            else:
                if node == node.parent.right:  # Triangle-shape
                    node = node.parent
                    _left_rotate(node)
                node.parent.make_black()
                node.parent.parent.make_red()
                _right_rotate(node.parent.parent)
        else:  # node's parent is a right-child
            uncle = node.parent.parent.left
            if uncle and uncle.is_red:
                node.parent.make_black
                uncle.make_black()
                node.parent.parent.make_red()
                node = node.parent.parent
            else:
                if node == node.parent.left:  # Triangle-shape
                    node = node.parent
                    _right_rotate(node)
                node.parent.make_black()
                node.parent.parent.make_red()
                _left_rotate(node.parent.parent)
    _root(node).make_black()

def move_clip_down(base_node, clip, move_left):
    """
    Moves a given clip to a lower node the the redblack search/sort tree and rebalances.
    Returns true if the tree changes as a result of the clip moving.
    """
    if move_left and base_node.left:
        base_node.left.pending_clips.add(clip)
        return False
    elif (not move_left) and base_node.right:
        base_node.right.pending_clips.add(clip)
        return False
    else:
        # We have to add a new node to the tree!
        new_node = SortTree(
            experiment_name=base_node.experiment_name,
            is_red=True,
            parent=base_node,
        )
        new_node.save()
        new_node.bound_clips.add(clip)
        if move_left:
            base_node.left = new_node
        else:
            base_node.right = new_node
        base_node.save()

        _rebalance_tree(new_node)

        return True
