import { Module } from '@nestjs/common';
import { AuthModule as BetterAuthModule } from '@thallesp/nestjs-better-auth';
import { auth } from './auth.config.js';

@Module({
  imports: [
    BetterAuthModule.forRoot(auth),
  ],
  exports: [BetterAuthModule]
})
export class AuthModule {}
