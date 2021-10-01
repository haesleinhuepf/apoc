import sklearn
import numpy as np


class DecisionTreeClassifierToOpenCLConverter:
    def __init__(self, decistion_tree_classifier):
        self.n_nodes = decistion_tree_classifier.tree_.node_count
        self.children_left = decistion_tree_classifier.tree_.children_left
        self.children_right = decistion_tree_classifier.tree_.children_right
        self.feature = decistion_tree_classifier.tree_.feature
        self.threshold = decistion_tree_classifier.tree_.threshold
        self.class_weight = decistion_tree_classifier.tree_.value

        self.node_depth = np.zeros(shape=self.n_nodes, dtype=np.int64)
        self.is_leaves = np.zeros(shape=self.n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            self.node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = self.children_left[node_id] != self.children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((self.children_left[node_id], depth + 1))
                stack.append((self.children_right[node_id], depth + 1))
            else:
                self.is_leaves[node_id] = True

    def to_opencl(self, i=0):
        if self.is_leaves[i]:
            output = ""
            for j, w in enumerate(self.class_weight[i][0]):
                if w > 0:
                    output += (self.node_depth[i] * " ") + "s" + str(j) + "+=" + str(w) + ";\n"
            return output
        else:
            output = "{space}if(i{feature}<{threshold})".format(
                space=self.node_depth[i] * " ",
                feature=self.feature[i],
                threshold=np.float32(self.threshold[i])) + '{\n'
            output += self.to_opencl(self.children_left[i])
            output += "{space}".format(space=self.node_depth[i] * " ") + '} else {\n'
            output += self.to_opencl(self.children_right[i])
            output += "{space}".format(space=self.node_depth[i] * " ") + '}\n'
            return output


def _ocl_header(num_inputs, num_classes):
    output = "__kernel void predict ("
    for i in range(0, num_inputs):
        output += "IMAGE_in" + str(i) + "_TYPE in" + str(i) + ", "

    output += "IMAGE_out_TYPE out) {\n"
    output += " sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
    output += " const int x = get_global_id(0);\n"
    output += " const int y = get_global_id(1);\n"
    output += " const int z = get_global_id(2);\n"
    for i in range(0, num_inputs):
        output += " float i" + str(i) + " = READ_IMAGE(in" + str(i) + ", sampler, POS_in" + str(
            i) + "_INSTANCE(x,y,z,0)).x;\n"

    for i in range(0, num_classes):
        output += " float s" + str(i) + "=0;\n"
    return output


def _ocl_footer(num_classes):
    output = " float max_s=s0;\n"
    output += " int cls=1;\n"
    for i in range(0, num_classes - 1):
        output += " if (max_s < s" + str(i + 1) + ") {\n  max_s = s" + str(i + 1) + ";\n  cls=" + str(i + 2) + ";\n }\n"

    output += " WRITE_IMAGE (out, POS_out_INSTANCE(x,y,z,0), cls);\n}\n"
    return output


def RFC_to_OCL(random_forest_classifier):
    """
    Converte a scikit-learn RandomForestClassifier to OpenCL code that mimiks the original
    
    Parameters
    ----------
    random_forest_classifier

    Returns
    -------

    """
    trees = random_forest_classifier.estimators_
    num_classes = random_forest_classifier.n_classes_
    num_inputs = random_forest_classifier.n_features_

    output = _ocl_header(num_inputs, num_classes)
    for tree in trees:
        output += DecisionTreeClassifierToOpenCLConverter(tree).to_opencl()
    output += _ocl_footer(num_classes)

    return output

