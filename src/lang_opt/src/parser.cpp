#include "parser.hpp"
#include <pybind11/stl.h>

namespace lang_opt {

Parser::Parser(const std::string& input_text) : position(0) {
    // Tokenize the input
    Lexer lexer(input_text);
    tokens = lexer.tokenize();
    
    if (!tokens.empty()) {
        current_token = tokens[0];
    } else {
        current_token = std::nullopt;
    }
    
    // Import Python AST node classes
    try {
        ast_nodes_module = py::module_::import("src.lang.ast_nodes");
        NumberNode = ast_nodes_module.attr("NumberNode");
        BooleanNode = ast_nodes_module.attr("BooleanNode");
        VariableNode = ast_nodes_module.attr("VariableNode");
        LambdaNode = ast_nodes_module.attr("LambdaNode");
        ApplicationNode = ast_nodes_module.attr("ApplicationNode");
        ListNode = ast_nodes_module.attr("ListNode");
        IfNode = ast_nodes_module.attr("IfNode");
    } catch (const py::error_already_set& e) {
        throw ParseError("Failed to import AST nodes: " + std::string(e.what()));
    }
}

void Parser::advance() {
    position++;
    if (position < tokens.size()) {
        current_token = tokens[position];
    } else {
        current_token = std::nullopt;
    }
}

std::optional<Token> Parser::peek(size_t offset) const {
    size_t peek_pos = position + offset;
    if (peek_pos < tokens.size()) {
        return tokens[peek_pos];
    }
    return std::nullopt;
}

Token Parser::expect(TokenType token_type) {
    if (!current_token || current_token->type != token_type) {
        throw ParseError("Expected token type mismatch", current_token);
    }
    
    Token token = *current_token;
    advance();
    return token;
}

py::object Parser::parse() {
    if (!current_token || current_token->type == TokenType::EOF_TOKEN) {
        throw ParseError("Empty input");
    }
    
    py::object ast = parse_expression();
    
    // Ensure we've consumed all input (except EOF)
    if (current_token && current_token->type != TokenType::EOF_TOKEN) {
        throw ParseError("Unexpected token after expression", current_token);
    }
    
    return ast;
}

py::object Parser::parse_expression() {
    if (!current_token) {
        throw ParseError("Unexpected end of input");
    }
    
    Token token = *current_token;
    
    // Number literal
    if (token.type == TokenType::NUMBER) {
        advance();
        int value = std::stoi(token.value);
        return NumberNode(value);
    }
    
    // Boolean literal
    else if (token.type == TokenType::BOOLEAN) {
        advance();
        bool value = (token.value == "true");
        return BooleanNode(value);
    }
    
    // Variable/identifier
    else if (token.type == TokenType::IDENT) {
        advance();
        return VariableNode(token.value);
    }
    
    // List literal
    else if (token.type == TokenType::LBRACKET) {
        return parse_list();
    }
    
    // S-expression (lambda, application, or special form)
    else if (token.type == TokenType::LPAREN) {
        return parse_s_expression();
    }
    
    else {
        throw ParseError("Unexpected token", token);
    }
}

py::object Parser::parse_list() {
    expect(TokenType::LBRACKET);
    
    py::list elements;
    while (current_token && current_token->type != TokenType::RBRACKET) {
        elements.append(parse_expression());
    }
    
    expect(TokenType::RBRACKET);
    return ListNode(elements);
}

py::object Parser::parse_s_expression() {
    expect(TokenType::LPAREN);
    
    if (!current_token || current_token->type == TokenType::RPAREN) {
        throw ParseError("Empty S-expression");
    }
    
    // Check for lambda
    if (current_token->type == TokenType::LAMBDA) {
        return parse_lambda();
    }
    
    // Check for special forms
    if (current_token->type == TokenType::IDENT) {
        if (current_token->value == "if") {
            return parse_if();
        }
    }
    
    // Otherwise, it's a function application
    return parse_application();
}

py::object Parser::parse_lambda() {
    expect(TokenType::LAMBDA);
    
    // Get parameter name
    if (!current_token || current_token->type != TokenType::IDENT) {
        throw ParseError("Lambda requires a parameter name");
    }
    std::string param = current_token->value;
    advance();
    
    // Parse body - exactly one expression
    if (!current_token || current_token->type == TokenType::RPAREN) {
        throw ParseError("Lambda requires a body expression");
    }
    
    py::object body = parse_expression();
    
    expect(TokenType::RPAREN);
    
    return LambdaNode(param, body);
}

py::object Parser::parse_if() {
    expect(TokenType::IDENT);  // consume 'if'
    
    // Parse condition
    if (!current_token || current_token->type == TokenType::RPAREN) {
        throw ParseError("If requires a condition");
    }
    py::object condition = parse_expression();
    
    // Parse then branch
    if (!current_token || current_token->type == TokenType::RPAREN) {
        throw ParseError("If requires a then expression");
    }
    py::object then_expr = parse_expression();
    
    // Parse else branch
    if (!current_token || current_token->type == TokenType::RPAREN) {
        throw ParseError("If requires an else expression");
    }
    py::object else_expr = parse_expression();
    
    expect(TokenType::RPAREN);
    
    return IfNode(condition, then_expr, else_expr);
}

py::object Parser::parse_application() {
    // Parse the function expression
    py::object function = parse_expression();
    
    // Parse arguments
    py::list arguments;
    while (current_token && current_token->type != TokenType::RPAREN) {
        arguments.append(parse_expression());
    }
    
    expect(TokenType::RPAREN);
    
    return ApplicationNode(function, arguments);
}

py::object parse(const std::string& input_text) {
    Parser parser(input_text);
    return parser.parse();
}

} // namespace lang_opt

