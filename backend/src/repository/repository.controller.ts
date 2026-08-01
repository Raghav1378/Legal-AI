import { Controller, Get, Patch, Param, Body } from '@nestjs/common';
import { RepositoryService } from './repository.service.js';
import { Roles } from '../auth/roles.decorator.js';
import { Role } from '@prisma/client';

@Controller('repository-sources')
@Roles(Role.SUPER_ADMIN)
export class RepositoryController {
  constructor(private repositoryService: RepositoryService) {}

  @Get()
  getSources() {
    return this.repositoryService.getAllSources();
  }

  @Get('seed')
  @Roles(Role.SUPER_ADMIN)
  seed() {
    return this.repositoryService.seed();
  }

  @Patch(':id')
  toggleSource(
    @Param('id') id: string,
    @Body() body: { enabled: boolean },
  ) {
    return this.repositoryService.toggleSource(id, body.enabled);
  }
}
