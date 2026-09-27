package com.example.users;

import java.util.Optional;

public class UserController {
    private final UserService userService;

    public UserController(UserService userService) {
        this.userService = userService;
    }

    public Optional<User> getUser(long id) {
        return userService.findUser(id);
    }
}
