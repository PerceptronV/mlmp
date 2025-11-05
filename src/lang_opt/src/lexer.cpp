#include "lexer.hpp"
#include <cctype>
#include <unordered_set>

namespace lang_opt {

Lexer::Lexer(std::string text) : input(std::move(text)), position(0) {
    current_char = input.empty() ? std::nullopt : std::optional<char>(input[0]);
}

void Lexer::advance() {
    position++;
    if (position < input.size()) {
        current_char = input[position];
    } else {
        current_char = std::nullopt;
    }
}

std::optional<char> Lexer::peek(size_t offset) const {
    size_t peek_pos = position + offset;
    if (peek_pos < input.size()) {
        return input[peek_pos];
    }
    return std::nullopt;
}

void Lexer::skip_whitespace() {
    while (current_char && std::isspace(*current_char)) {
        advance();
    }
}

void Lexer::skip_comment() {
    if (current_char == '#') {
        while (current_char && *current_char != '\n') {
            advance();
        }
        if (current_char == '\n') {
            advance();
        }
    }
}

std::string Lexer::read_number() {
    std::string result;
    while (current_char && std::isdigit(*current_char)) {
        result += *current_char;
        advance();
    }
    return result;
}

std::string Lexer::read_identifier() {
    std::string result;
    
    static const std::unordered_set<char> special_chars = {
        '+', '-', '*', '/', '%', '<', '>', '=', '_', '?', '!'
    };
    
    while (current_char) {
        if (std::isalnum(*current_char) || special_chars.count(*current_char)) {
            result += *current_char;
            advance();
        } else {
            break;
        }
    }
    
    return result;
}

Token Lexer::get_next_token() {
    while (current_char) {
        // Skip whitespace
        if (std::isspace(*current_char)) {
            skip_whitespace();
            continue;
        }
        
        // Skip comments
        if (*current_char == '#') {
            skip_comment();
            continue;
        }
        
        size_t token_pos = position;
        
        // Lambda symbol (λ)
        // UTF-8 encoding of λ is 0xCE 0xBB
        if (*current_char == '\xCE' && peek() == '\xBB') {
            advance();
            advance();
            return Token(TokenType::LAMBDA, "λ", token_pos);
        }
        
        // Single character tokens
        if (*current_char == '(') {
            advance();
            return Token(TokenType::LPAREN, "(", token_pos);
        }
        
        if (*current_char == ')') {
            advance();
            return Token(TokenType::RPAREN, ")", token_pos);
        }
        
        if (*current_char == '[') {
            advance();
            return Token(TokenType::LBRACKET, "[", token_pos);
        }
        
        if (*current_char == ']') {
            advance();
            return Token(TokenType::RBRACKET, "]", token_pos);
        }
        
        // Numbers
        if (std::isdigit(*current_char)) {
            std::string number = read_number();
            return Token(TokenType::NUMBER, number, token_pos);
        }
        
        // Identifiers, keywords, and operators
        if (std::isalpha(*current_char) || *current_char == '_' ||
            *current_char == '+' || *current_char == '-' || *current_char == '*' ||
            *current_char == '/' || *current_char == '%' || *current_char == '<' ||
            *current_char == '>' || *current_char == '=' || *current_char == '?' ||
            *current_char == '!') {
            std::string ident = read_identifier();
            
            // Check for boolean keywords
            if (ident == "true" || ident == "false") {
                return Token(TokenType::BOOLEAN, ident, token_pos);
            }
            
            return Token(TokenType::IDENT, ident, token_pos);
        }
        
        // Unknown character
        throw LexerError(std::string("Unexpected character: '") + *current_char + "'", position);
    }
    
    // End of input
    return Token(TokenType::EOF_TOKEN, "", position);
}

std::vector<Token> Lexer::tokenize() {
    std::vector<Token> tokens;
    
    while (true) {
        Token token = get_next_token();
        tokens.push_back(token);
        
        if (token.type == TokenType::EOF_TOKEN) {
            break;
        }
    }
    
    return tokens;
}

std::vector<Token> tokenize(const std::string& input_text) {
    Lexer lexer(input_text);
    return lexer.tokenize();
}

} // namespace lang_opt

