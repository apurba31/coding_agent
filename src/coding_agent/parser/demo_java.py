import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

JAVA = Language(tsjava.language())

parser = Parser(JAVA)

source = b"""
package com.example.user;

import java.util.List;

public class UserService {

    private final UserRepository repository;

    public UserService(UserRepository repository) {
        this.repository = repository;
    }

    public User findUser(Long id) {
        return repository.findById(id);
    }
}
"""
tree = parser.parse(source)
print(tree.root_node)
