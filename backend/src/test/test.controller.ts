import { Controller, Get, Req } from '@nestjs/common';
import { Roles } from '../auth/roles.decorator.js';
import { Role } from '@prisma/client';

@Controller('test')
export class TestController {
  
  @Get('user')
  @Roles(Role.USER)
  userRoute(@Req() req) {
    return { message: 'User route accessed', user: req.user };
  }

  @Get('admin')
  @Roles(Role.ADMIN)
  adminRoute() {
    return { message: 'Admin route accessed' };
  }
}
