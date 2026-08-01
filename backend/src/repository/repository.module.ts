import { Module } from '@nestjs/common';
import { RepositoryService } from './repository.service.js';
import { RepositoryController } from './repository.controller.js';

@Module({
  providers: [RepositoryService],
  controllers: [RepositoryController]
})
export class RepositoryModule {}
