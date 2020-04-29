
#
# Main component of TF Micro project 'mnist'.
#

idf_component_register(
    SRCS esp/main.cc main_functions.cc mnist_model_data.cc output_handler.cc constants.cc number_data.cc
    INCLUDE_DIRS "")
