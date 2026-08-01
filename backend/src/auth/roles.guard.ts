import { Injectable, CanActivate, ExecutionContext } from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { Role } from '@prisma/client';
import { auth } from '../auth/auth.config.js';

@Injectable()
export class BetterRolesGuard implements CanActivate {
  constructor(private reflector: Reflector) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const requiredRoles = this.reflector.getAllAndOverride<Role[]>('roles', [
      context.getHandler(),
      context.getClass(),
    ]);

    const request = context.switchToHttp().getRequest();

    // Fetch session from Better Auth
    const session = await auth.api.getSession({
      headers: new Headers(request.headers),
    });

    if (!session?.user) {
      return false;
    }

    // Attach user to request so controller can access req.user
    request.user = session.user;

    if (!requiredRoles) {
      return true;
    }

    return requiredRoles.includes(session.user.role as Role);
  }
}