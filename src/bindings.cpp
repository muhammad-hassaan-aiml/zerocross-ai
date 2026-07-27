#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Essential for std::vector <-> Python list conversion
#include "game_state.h"
#include "mcts.h"

namespace py = pybind11;

PYBIND11_MODULE(zerocross_engine, m) {
    m.doc() = "ZeroCross Ultimate Tic-Tac-Toe C++ Engine";

    // Expose GameState
    py::class_<GameState>(m, "GameState")
        .def(py::init<>())
        .def("encode", &GameState::encode)
        .def("legal_mask", &GameState::legal_mask)
        .def("play", &GameState::play)
        .def("is_terminal", &GameState::is_terminal)
        .def("get_current_player", &GameState::get_current_player) // Phase 5 addition
        .def("get_winner", &GameState::get_winner)                 // Phase 5 addition
        .def_static("from_array", &GameState::from_array);

    // Expose MCTSTree
    py::class_<MCTSTree>(m, "MCTSTree")
        .def(py::init<const GameState&, bool>(), py::arg("root_state"), py::arg("add_noise") = true)
        .def("request_leaf", &MCTSTree::request_leaf)
        .def("submit_result", &MCTSTree::submit_result)
        .def("is_done", &MCTSTree::is_done)
        .def("root_policy", &MCTSTree::root_policy)
        .def("advance", &MCTSTree::advance);
}