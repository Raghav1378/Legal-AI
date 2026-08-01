import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service.js';
import { AiService } from '../ai/ai.service.js';
import { MessageRole, AgentStatus, Role } from '@prisma/client';

@Injectable()
export class ChatService {
  constructor(
    private prisma: PrismaService,
    private aiService: AiService,
  ) {}

  async handleQuery(
    userId: string,
    chatId: string | undefined,
    query: string,
  ) {

    let chat;

    if (!chatId) {
      chat = await this.prisma.chat.create({
        data: {
          userId,
          title: query.substring(0, 30),
        },
      });
    } else {
      chat = await this.prisma.chat.findUnique({
        where: { id: chatId },
      });
    }

    if (!chat) {
      throw new Error('Chat not found');
    }

    await this.prisma.message.create({
      data: {
        chatId: chat.id,
        role: MessageRole.USER,
        content: query,
      },
    });

    const startTime = Date.now();

    const aiResult = await this.aiService.generateLegalResponse(
      query,
      chat.id,
    );

    const responseTime = Date.now() - startTime;

    const structured = aiResult.structuredResponse;
    console.log("AI RESULT FULL:", aiResult);
    console.log("STRUCTURED:", structured);

    const assistantMessage = await this.prisma.message.create({
      data: {
        chatId: chat.id,
        role: MessageRole.ASSISTANT,
        content: structured.issue_summary,
        structuredResponse: structured,
        confidenceScore:
          structured.confidence_score > 1
            ? structured.confidence_score
            : structured.confidence_score * 100,
        conflictsDetected: structured.conflicts_detected,
      },
    });

    if (structured.citations?.length) {
      for (const citation of structured.citations) {
        await this.prisma.citation.create({
          data: {
            messageId: assistantMessage.id,
            title:
            citation.title ||
                citation.citation_reference ||
              "Unknown Citation",
            court: citation.court ?? null,
            year:
              typeof citation.year === "number"
                ? citation.year
                : null,
            source:
              citation.source ||
              citation.citation_reference ||
              "General Law",
            url: citation.url || citation.source_url || null,
          },
        });
      }
    }

    if (aiResult.agentLogs?.length) {
      for (const log of aiResult.agentLogs) {
        await this.prisma.agentExecutionLog.create({
          data: {
            chatId: chat.id,
            agentName: log.agentName,
            executionTimeMs: log.executionTimeMs,
            status:
              log.status === 'FAILED'
                ? AgentStatus.FAILED
                : AgentStatus.SUCCESS,
            confidenceScore: structured.confidence_score,
            conflictsDetected: structured.conflicts_detected,
          },
        });
      }
    }

    if (aiResult.totalExecutionTimeMs) {
      await this.prisma.agentExecutionLog.create({
        data: {
          chatId: chat.id,
          agentName: 'TOTAL_EXECUTION',
          executionTimeMs: aiResult.totalExecutionTimeMs,
          status: AgentStatus.SUCCESS,
          confidenceScore: structured.confidence_score,
          conflictsDetected: structured.conflicts_detected,
        },
      });
    }

    await this.prisma.auditLog.create({
      data: {
        userId,
        action: 'SUBMIT_QUERY',
        metadata: {
          chatId: chat.id,
          query,
          backendResponseTimeMs: responseTime,
          aiExecutionTimeMs: aiResult.totalExecutionTimeMs,
        },
      },
    });

    return {
      chatId: chat.id,
      messageId: assistantMessage.id,
      isBookmarked: assistantMessage.isBookmarked,
      backendResponseTimeMs: responseTime,
      aiExecutionTimeMs: aiResult.totalExecutionTimeMs,
      ...structured,
    };
  }

  async getChatById(
    chatId: string,
    userId: string,
    role: Role,
  ) {
    const chat = await this.prisma.chat.findUnique({
      where: { id: chatId },
      include: {
        messages: {
          include: {
            citations: true,
          },
          orderBy: {
            createdAt: 'asc',
          },
        },
      },
    });

    if (!chat) {
      throw new Error('Chat not found');
    }

    if (role === Role.USER && chat.userId !== userId) {
      throw new Error('Unauthorized');
    }

    return chat;
  }

  async getUserChats(userId: string) {
    return this.prisma.chat.findMany({
      where: { userId },
      orderBy: { createdAt: 'desc' },
      select: {
        id: true,
        title: true,
        createdAt: true,
      },
    });
  }

  async toggleBookmark(messageId: string) {
    const message = await this.prisma.message.findUnique({
      where: { id: messageId },
    });

    if (!message) {
      throw new Error('Message not found');
    }

    return this.prisma.message.update({
      where: { id: messageId },
      data: {
        isBookmarked: !message.isBookmarked,
      },
    });
  }
}
