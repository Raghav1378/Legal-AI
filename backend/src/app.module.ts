import { Module } from '@nestjs/common';
import { APP_GUARD } from '@nestjs/core';
import { BetterRolesGuard } from './auth/roles.guard.js';

import { AppController } from './app.controller.js';
import { AppService } from './app.service.js';
import { PrismaModule } from './prisma/prisma.module.js';
import { AuthModule } from './auth/auth.module.js';
import { UsersModule } from './users/users.module.js';
import { TestController } from './test/test.controller.js';
import { ChatModule } from './chat/chat.module.js';
import { AiModule } from './ai/ai.module.js';
import { AdminController } from './admin/admin.controller.js';
import { AdminService } from './admin/admin.service.js';
import { RepositoryModule } from './repository/repository.module.js';

@Module({
  imports: [PrismaModule, AuthModule, UsersModule, ChatModule, AiModule, RepositoryModule],
  controllers: [AppController, TestController, AdminController],
  providers: [
    AppService, 
    AdminService,
    {
      provide: APP_GUARD,
      useClass: BetterRolesGuard,
    },
  ],

})
export class AppModule {}
