import re
import json
from copy import deepcopy


class DialogueManager:
    def __init__(self, system_prompt=None, log_path=None):
        self.path = log_path
        self.system_prompt = system_prompt
        self._base = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

        self._data = deepcopy(self._base.copy())

    def data(self): return deepcopy(self._data)
    def add(self, messages): self._data.extend(messages); self.save()
    def reset(self): self._data = deepcopy(self._base); self.save()

    def save(self):
        if self.path is not None:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=1, ensure_ascii=False)


class Base:
    def __init__(self):
        ...
        
    def format_question(self, q): return (q or "").strip()
    def extract_response(self, o): return {"answer": o, "citations": None}
    def format_facts(self, f): return ""
    def format_citations(self, c): return ""
    
    def format_user(self, question, facts=None):
        question = self.format_question(question)
        user = {"role":"user", "content": f"{question}"} ## for sft 
        meta = {"role":"user", "content": f"{question}"} ## for eval and history
        return user, meta

    def format_assistant(self, content, citations=None):
        assistant = {"role":"assistant", "content": content}                     ## for sft 
        meta =      {"role":"assistant", "content": content, "citations": None}  ## for eval and history
        return assistant, meta
    

class Context(Base):

    def format_facts(self, facts):
        context_lines = ["Facts:"]
        for i, f in enumerate(facts, 1):
            text = f['title']
            context_lines.append(f"[{i}] {text}")
        context_block = "\n".join(context_lines)
        return context_block

    def format_user(self, question, facts):
        question = self.format_question(question)
        context = self.format_facts(facts)
        user = {"role":"user", "content": f"{question}\n{context}".strip()} ## for sft 
        meta = {"role":"user", "content": f"{question}", "context":facts}   ## for eval
        return user, meta
    

class ContextCitations(Base):

    def format_facts(self, facts):
        context_lines = ["Facts:"]
        for i, f in enumerate(facts, 1):
            text = f['title']
            context_lines.append(f"[{i}] {text}")
        context_block = "\n".join(context_lines)
        return context_block
    
    def format_citations(self, c): 
        return c

    def extract_json(self, text: str):
        """Extract first valid JSON from messy LLM output."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        text = re.sub(r"```(?:json|python|javascript)?|```", "", text)
        candidates = re.findall(r"\{.*?\}", text, flags=re.S)
        for c in candidates:
            try:
                return json.loads(c)
            except json.JSONDecodeError:
                c = re.sub(r",\s*([}\]])", r"\1", c)
                try:
                    return json.loads(c)
                except json.JSONDecodeError:
                    continue
        raise ValueError("No valid JSON found.")

    def format_user(self, question, facts):
        question = self.format_question(question)
        context = self.format_facts(facts)
        facts = json.dumps([{"fid":f['id'], "text":f['title']} for f in facts], ensure_ascii=False)
        user = {"role":"user", "content": f"คำถาม:\n{question}\n{context}".strip()} ## for sft 
        meta = {"role":"user", "content": f"{question}", "context":facts}   ## for eval
        return user, meta

    def format_assistant(self, content, citations):
        citations = self.format_citations(citations)
        ## for sft (LLama factory)
        ## ทำไมต้อง ่json.dumps; เพราะตอน last turn ต้องใช้
        assistant = {
            "role":"assistant", 
            "content": json.dumps({
                "answer": content, 
                "citations": citations
            } , ensure_ascii=False
        )}  
        ## for the next history: store content (llm response, only answer) 
        ## for eval and history
        meta = {"role":"assistant", "content": content, "llm_citations": citations}
        return assistant, meta

    def extract_response(self, llm_response):
        try:
            out = self.extract_json(llm_response)
            ans = out.get("answer", out)
            citations = out.get('citations', []) 
            return {"answer": ans, "citations": citations, "llm_response": llm_response}
        except:
            return {"answer": llm_response, "citations": [], "llm_response": llm_response}
        
    

class FormatPrompt:
    def __init__(self, prompt_template):

        ## Select prompt template
        if prompt_template == "base":
            self.template = Base()
        elif prompt_template == "context":
            self.template = Context()
        elif prompt_template == "contextcitations":
            self.template = ContextCitations()
        else:
            raise ValueError(f"Unknown template: {prompt_template}")

    
    def format_user(self, question, facts):
        return self.template.format_user(question=question, facts=facts)

    def format_assistant(self, assistant, citations):
        return self.template.format_assistant(assistant, citations)

    def extract_response(self, text):
        return self.template.extract_response(text)
    
    def get_system_prompt(self):
        return self.template.system_prompt
    