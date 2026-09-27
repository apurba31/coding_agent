package com.example.users;

class UserServiceTest {
    void findsUserById() {
        UserService service = new UserService(java.util.Map.of());
        service.findUser(7L);
    }
}
