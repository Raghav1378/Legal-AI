import { Controller, Get } from '@nestjs/common';
import { Roles } from '../auth/roles.decorator.js';
import { Role } from '@prisma/client';
import { AdminService } from './admin.service.js';

@Controller('admin')
@Roles(Role.ADMIN, Role.SUPER_ADMIN)
export class AdminController {
  constructor(private adminService: AdminService) {}

  @Get('queries')
  getAllQueries() {
    return this.adminService.getAllQueries();
  }

  @Get('analytics')
  getAnalytics() {
    return this.adminService.getAnalytics();
  }

  @Get('audit-logs')
  getAuditLogs() {
    return this.adminService.getAuditLogs();
  }

  @Get('agent-logs')
  getAgentLogs() {
    return this.adminService.getAgentLogs();
  }
}
