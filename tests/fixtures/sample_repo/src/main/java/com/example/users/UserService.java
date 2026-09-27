package com.example.users;

import java.util.Map;
import java.util.Optional;

public class UserService {
    private final Map<Long, User> users;

    public UserService(Map<Long, User> users) {
        this.users = users;
    }

    /** Returns the user matching the requested identifier. */
    public Optional<User> findUser(long id) {
        return Optional.ofNullable(users.get(id));
    }
}
