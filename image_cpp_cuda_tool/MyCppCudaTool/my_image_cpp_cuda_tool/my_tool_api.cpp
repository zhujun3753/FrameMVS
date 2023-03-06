#include <torch/serialize/tensor.h>
#include <torch/extension.h>
// #include <torch/torch.h>


#include "my_tool_gpu.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    //* 第一个参数是python调用的函数名，第二个参数是c++函数地址，第三个变量是函数description
    py::class_<Params>(m, "Params")  
        .def(py::init())
        // .def("setName", &student::setName)  
        .def_readwrite("b",         &Params::b)
        .def_readwrite("h",         &Params::h)
        .def_readwrite("w",         &Params::w)
        .def_readwrite("p",         &Params::p)
        .def_readwrite("npts",      &Params::npts)
        .def_readwrite("ndepth",    &Params::ndepth)
        .def_readwrite("nsrc",      &Params::nsrc)
        .def_readwrite("height",    &Params::height)
        .def_readwrite("stage_nsrc",&Params::stage_nsrc)
        .def_readwrite("edge_thred",&Params::edge_thred)
        .def_readwrite("patch_size",&Params::patch_size)
        .def_readwrite("ncc_mode",  &Params::ncc_mode)
        .def_readwrite("skip_lines",&Params::skip_lines)  //* 关键点检测时跳过的行 
        .def_readwrite("kpts_detect_edge_thred",  &Params::kpts_detect_edge_thred)
        .def_readwrite("kpts_suppresion_radius",  &Params::kpts_suppresion_radius)
        .def_readwrite("width",     &Params::width)
        .def_readwrite("kpt_radius",&Params::kpt_radius); //* 按行半径kpt_radius个像素内edge最大的为关键点


    m.def("struct_test", &struct_test);
    m.def("image_seg", &image_seg_cpp, "image_seg_cpp");
    m.def("kpts_selector", &kpts_selector_cpp, "kpts_selector_cpp");
    m.def("kpts_depth_pred", &kpts_depth_pred_cpp, "kpts_depth_pred");
    m.def("kpts_depth_pred_struct", &kpts_depth_pred_struct_cpp, "kpts_depth_pred_struct");
    m.def("kpts_depth_ncc", &kpts_depth_ncc_cpp, "kpts_depth_ncc");
    m.def("kpts_detector", &kpts_detector_cpp, "kpts_detector");





}
