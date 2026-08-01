import { jwtDecode } from "jwt-decode";

export const getUserRole = () => {
  const token = localStorage.getItem("token");
  if (!token) return null;

  const decoded: any = jwtDecode(token);
  return decoded.role;
};