const BASE_URL = "http://localhost:3001";

export const apiFetch = async (
  endpoint: string,
  options: RequestInit = {}
) => {
  const res = await fetch(`${BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    credentials: "include",
  });


  if (!res.ok) {
    throw new Error("API error");
  }

  return res.json();
};