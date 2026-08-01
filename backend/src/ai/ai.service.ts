import { Injectable, HttpException } from '@nestjs/common';
import axios from 'axios';

@Injectable()
export class AiService {

  private FASTAPI_URL = process.env.FASTAPI_URL || 'http://127.0.0.1:8000/query';

  async generateLegalResponse(query: string, chatId?: string) {
    try {
      const response = await axios.post(
        this.FASTAPI_URL,
        {
          chat_id: chatId || `session-${Date.now()}`,
          query: query,
        },
        {
          timeout: 160000,
          headers: {
            'Content-Type': 'application/json',
          },
        },
      );

      const data = response.data;

      console.log('RAW FASTAPI RESPONSE:', JSON.stringify(data, null, 2));

      return {
        structuredResponse: data.structuredResponse ?? data,
        agentLogs: data.agentLogs || [],
        totalExecutionTimeMs: data.totalExecutionTimeMs || null,
      };

    } catch (error: any) {
      console.error(
        'FastAPI Error:',
        error?.response?.data || error.message,
      );

      throw new HttpException(
        error?.response?.data?.detail || 'AI Engine Failed',
        error?.response?.status || 500,
      );
    }
  }
}
