import { createAuthClient } from "better-auth/react";

export const authClient = createAuthClient({
    baseURL: "http://localhost:3001",
    user: {
        additionalFields: {
            role: {
                type: "string",
            },
        },
    },
});

