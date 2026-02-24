from typing import List

class ResearchPlanner:
    @staticmethod
    def plan(query: str) -> List[str]:
        lower_query = query.lower()
        sub_questions = []
        
        if "bail" in lower_query or "arrest" in lower_query:
            sub_questions.append("What are the legal provisions for anticipatory bail?")
            sub_questions.append("What is the landmark Supreme Court judgment on Section 438 CrPC?")
            
        if "murder" in lower_query or "302" in lower_query:
            sub_questions.append("What is the punishment for murder under IPC Section 302?")
            sub_questions.append("What are the exceptions to Section 302 IPC?")

        if not sub_questions:
            sub_questions.append(f"General research on: {query}")
            
        return sub_questions
