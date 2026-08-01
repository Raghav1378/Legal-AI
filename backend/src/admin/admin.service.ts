import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service.js';
import { MessageRole, AgentStatus } from '@prisma/client';

@Injectable()
export class AdminService {
  constructor(private prisma: PrismaService) {}

  async getAllQueries() {
    return this.prisma.chat.findMany({
      include: {
        user: true,
        messages: {
          where: { role: MessageRole.ASSISTANT },
          select: {
            confidenceScore: true,
            conflictsDetected: true,
            createdAt: true,
          },
        },
      },
      orderBy: {
        createdAt: 'desc',
      },
    });
  }

  async getAnalytics() {
    const totalQueries = await this.prisma.message.count({
      where: { role: MessageRole.ASSISTANT },
    });

    const avgConfidence = await this.prisma.message.aggregate({
      where: { role: MessageRole.ASSISTANT },
      _avg: { confidenceScore: true },
    });

    const conflictCount = await this.prisma.message.count({
      where: { conflictsDetected: true },
    });

    const agentFailures = await this.prisma.agentExecutionLog.count({
      where: { status: AgentStatus.FAILED },
    });

    return {
      totalQueries,
      avgConfidence: avgConfidence._avg.confidenceScore || 0,
      conflictCount,
      agentFailures,
    };
  }

  async getAuditLogs() {
    return this.prisma.auditLog.findMany({
      include: {
        user: true,
      },
      orderBy: {
        createdAt: 'desc',
      },
    });
  }

  async getAgentLogs() {
    return this.prisma.agentExecutionLog.findMany({
      orderBy: { createdAt: 'desc' },
    });
  }
}
