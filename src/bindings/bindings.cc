#include <core/mcts/model_info.h>
#include <core/state/serializer.h>
#include <core/state/state.h>
#include <pybind11/pybind11.h>

#include <ranges>

namespace py = pybind11;
using namespace alpack;

template <typename ArrayT>
constexpr auto bind_ndarray(py::module_& m, const char* name) -> void {
  py::class_<ArrayT>(m, name, py::buffer_protocol()).def_buffer([](ArrayT& arr) {
    using T = ArrayT::value_type;
    py::buffer_info info{};
    info.ptr = arr.data();
    info.itemsize = sizeof(T);
    info.format = py::format_descriptor<T>::format();
    info.ndim = arr.ndim;
    info.shape = arr.shape | std::ranges::to<std::vector<int64_t>>();
    info.strides = arr.strides | std::ranges::to<std::vector<int64_t>>();
    return info;
  });
}

PYBIND11_MODULE(alphapack, m) {
  // Vec3i
  py::class_<Vec3<int>>(m, "Vec3i")
    .def(py::init<int, int, int>(), py::arg("x"), py::arg("y"), py::arg("z"))
    .def_readwrite("x", &Vec3<int>::x)
    .def_readwrite("y", &Vec3<int>::y)
    .def_readwrite("z", &Vec3<int>::z);

  // Item
  py::class_<Item>(m, "Item")
    .def(py::init<>())
    .def_readwrite("shape", &Item::shape)
    .def_readwrite("placed", &Item::placed);

  // NdArray
  bind_ndarray<State::Array2D<int8_t>>(m, "Array2Dint8");
  bind_ndarray<State::Array2D<uint8_t>>(m, "Array2Duint8");
  bind_ndarray<State::Array2D<bool>>(m, "Array2Dbool");

  // State
  py::class_<State>(m, "State")
    .def(py::init<std::vector<Item>>(), py::arg("items"))
    .def_property_readonly("items", &State::items)
    .def_property_readonly("height_map", &State::height_map)
    .def_property_readonly("feasibility_info", &State::feasibility_info)
    .def_property_readonly("feasibility_mask", &State::feasibility_mask)
    .def_property_readonly("packing_efficiency", &State::packing_efficiency)
    .def_property_readonly("feasible_actions", &State::feasible_actions)
    .def_readonly_static("bin_length", &State::bin_length)
    .def_readonly_static("bin_height", &State::bin_height)
    .def_readonly_static("action_count", &State::action_count)
    .def_readonly_static("max_item_count", &State::max_item_count)
    .def("transition", &State::transition, py::arg("action_idx"))
    .def(
      py::pickle(
        [](const State& state) { return py::bytes(Serializer<State>::serialize(state)); },
        [](const py::bytes& bytes) { return Serializer<State>::unserialize(bytes); }
      )
    );

  // ModelInfo
  py::class_<ModelInfo>(m, "ModelInfo")
    .def_readonly_static("input_feature_count", &ModelInfo::input_feature_count)
    .def_readonly_static("additional_input_count", &ModelInfo::additional_input_count)
    .def_readonly_static("value_support_count", &ModelInfo::value_support_count);
}