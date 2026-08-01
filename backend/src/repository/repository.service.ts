import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service.js';

@Injectable()
export class RepositoryService {
  constructor(private prisma: PrismaService) {}

  async getAllSources() {
    return this.prisma.repositorySource.findMany({
      orderBy: { createdAt: 'desc' },
    });
  }

  async toggleSource(id: string, enabled: boolean) {
    return this.prisma.repositorySource.update({
      where: { id },
      data: { enabled },
    });
  }

  async seed() {
    return this.prisma.repositorySource.createMany({
      data: [
        { name: 'Supreme Court Judgments', enabled: true },
        { name: 'High Court Judgments', enabled: true },
        { name: 'Indian Penal Code', enabled: true },
      ],
    });
  }
}
