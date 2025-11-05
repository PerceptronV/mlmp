#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "lexer.hpp"
#include "parser.hpp"

namespace py = pybind11;
using namespace lang_opt;

PYBIND11_MODULE(lang_opt_native, m) {
    m.doc() = "High-performance C++ implementation of lexer and parser";

    // ========================================================================
    // Lexer bindings
    // ========================================================================
    
    py::enum_<TokenType>(m, "TokenType")
        .value("LAMBDA", TokenType::LAMBDA)
        .value("LPAREN", TokenType::LPAREN)
        .value("RPAREN", TokenType::RPAREN)
        .value("LBRACKET", TokenType::LBRACKET)
        .value("RBRACKET", TokenType::RBRACKET)
        .value("NUMBER", TokenType::NUMBER)
        .value("BOOLEAN", TokenType::BOOLEAN)
        .value("IDENT", TokenType::IDENT)
        .value("EOF_TOKEN", TokenType::EOF_TOKEN);
    
    py::class_<Token>(m, "Token")
        .def(py::init<TokenType, std::string, size_t>(),
             py::arg("type"), py::arg("value") = "", py::arg("position") = 0)
        .def_readwrite("type", &Token::type)
        .def_readwrite("value", &Token::value)
        .def_readwrite("position", &Token::position)
        .def("__repr__", [](const Token& t) {
            std::string type_name;
            switch (t.type) {
                case TokenType::LAMBDA: type_name = "LAMBDA"; break;
                case TokenType::LPAREN: type_name = "LPAREN"; break;
                case TokenType::RPAREN: type_name = "RPAREN"; break;
                case TokenType::LBRACKET: type_name = "LBRACKET"; break;
                case TokenType::RBRACKET: type_name = "RBRACKET"; break;
                case TokenType::NUMBER: type_name = "NUMBER"; break;
                case TokenType::BOOLEAN: type_name = "BOOLEAN"; break;
                case TokenType::IDENT: type_name = "IDENT"; break;
                case TokenType::EOF_TOKEN: type_name = "EOF"; break;
            }
            if (!t.value.empty()) {
                return type_name + "(" + t.value + ")";
            }
            return type_name;
        });
    
    py::class_<Lexer>(m, "Lexer")
        .def(py::init<std::string>())
        .def("get_next_token", &Lexer::get_next_token)
        .def("tokenize", &Lexer::tokenize);
    
    m.def("tokenize", &tokenize, "Tokenize input string into tokens",
          py::arg("input_text"));
    
    py::register_exception<LexerError>(m, "LexerError");
    
    // ========================================================================
    // Parser bindings
    // ========================================================================
    
    py::class_<Parser>(m, "Parser")
        .def(py::init<std::string>())
        .def("parse", &Parser::parse);
    
    m.def("parse", &parse, "Parse input string into AST",
          py::arg("input_text"));
    
    py::register_exception<ParseError>(m, "ParseError");
    
    // ========================================================================
    // Version info
    // ========================================================================
    
    m.attr("__version__") = "0.1.0";
}

