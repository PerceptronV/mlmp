#pragma once

#include "lexer.hpp"
#include <pybind11/pybind11.h>
#include <memory>
#include <stdexcept>

namespace py = pybind11;

namespace lang_opt {

class ParseError : public std::runtime_error {
public:
    std::optional<Token> token;
    
    ParseError(const std::string& message, const std::optional<Token>& tok = std::nullopt)
        : std::runtime_error(tok ? "Parse error at position " + std::to_string(tok->position) + ": " + message
                                 : "Parse error: " + message),
          token(tok) {}
};

class Parser {
private:
    std::vector<Token> tokens;
    size_t position;
    std::optional<Token> current_token;
    
    // Python AST module and node classes
    py::object ast_nodes_module;
    py::object NumberNode;
    py::object BooleanNode;
    py::object VariableNode;
    py::object LambdaNode;
    py::object ApplicationNode;
    py::object ListNode;
    py::object IfNode;

    void advance();
    std::optional<Token> peek(size_t offset = 1) const;
    Token expect(TokenType token_type);
    
    py::object parse_expression();
    py::object parse_list();
    py::object parse_s_expression();
    py::object parse_lambda();
    py::object parse_if();
    py::object parse_application();

public:
    explicit Parser(const std::string& input_text);
    
    py::object parse();
};

// Convenience function
py::object parse(const std::string& input_text);

} // namespace lang_opt

