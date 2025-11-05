#pragma once

#include <string>
#include <vector>
#include <optional>
#include <stdexcept>

namespace lang_opt {

enum class TokenType {
    LAMBDA,      // λ
    LPAREN,      // (
    RPAREN,      // )
    LBRACKET,    // [
    RBRACKET,    // ]
    NUMBER,      // Integer literals
    BOOLEAN,     // true, false
    IDENT,       // Variable names and function names
    EOF_TOKEN    // End of input
};

struct Token {
    TokenType type;
    std::string value;
    size_t position;

    Token(TokenType t, std::string v = "", size_t pos = 0)
        : type(t), value(std::move(v)), position(pos) {}
};

class LexerError : public std::runtime_error {
public:
    size_t position;
    
    LexerError(const std::string& message, size_t pos)
        : std::runtime_error("Lexer error at position " + std::to_string(pos) + ": " + message),
          position(pos) {}
};

class Lexer {
private:
    std::string input;
    size_t position;
    std::optional<char> current_char;

    void advance();
    std::optional<char> peek(size_t offset = 1) const;
    void skip_whitespace();
    void skip_comment();
    std::string read_number();
    std::string read_identifier();

public:
    explicit Lexer(std::string text);
    
    Token get_next_token();
    std::vector<Token> tokenize();
};

// Convenience function
std::vector<Token> tokenize(const std::string& input_text);

} // namespace lang_opt

