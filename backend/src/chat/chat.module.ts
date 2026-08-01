import { Module } from '@nestjs/common';
import { ChatService } from './chat.service.js';
import { ChatController } from './chat.controller.js';
import { AiModule } from '../ai/ai.module.js';


@Module({
  imports: [AiModule],
  providers: [ChatService],
  controllers: [ChatController]
})
export class ChatModule {}
