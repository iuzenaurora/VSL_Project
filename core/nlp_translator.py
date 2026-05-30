# -*- coding: utf-8 -*-
import os
import requests
import re
import google.generativeai as genai

# ==========================================
# CẤU HÌNH AI DỊCH THUẬT
# ==========================================
USE_GEMINI = False         # Đặt False để TẮT hoàn toàn Gemini
USE_LOCAL_OLLAMA = False   # Đặt True nếu bạn dùng phần mềm Ollama (chạy AI Offline)
OLLAMA_MODEL = "qwen2:0.5b" # Mô hình AI offline siêu nhẹ của Ollama

# Khởi tạo Gemini AI (Hãy điền API Key thật của bạn vào đây)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyD-Dlk1m0vZ3wgcm3Z-cW_WHkOZxFcPnyM")
gemini_model = None

if GEMINI_API_KEY and USE_GEMINI:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        # Tự động tìm model tốt nhất được hỗ trợ bởi phiên bản thư viện hiện tại
        selected_model = 'gemini-1.0-pro'
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'gemini-1.5-flash' in m.name:
                    selected_model = m.name
                    break
                elif 'gemini-pro' in m.name:
                    selected_model = m.name
        
        gemini_model = genai.GenerativeModel(selected_model)
        print(f"[Gemini Setup] Đã tự động kết nối với model: {selected_model}")
    except Exception as e:
        print(f"[Gemini Setup Error] Không thể nạp danh sách model: {e}")

class NLPTranslator:
    def __init__(self):
        self.buffer = []
        self.vocab_mapping = {
            "0 (số không)": "không",
            "tp. hồ chí minh": "TP. Hồ Chí Minh",
            "đồng nai": "Đồng Nai",
        }

    def add_gloss(self, word: str):
        w = word.lower()
        if not self.buffer or self.buffer[-1] != w:
            self.buffer.append(w)

    def _build_raw_sentence(self) -> str:
        processed = []
        spell_buf = ""

        for word in self.buffer:
            if len(word) == 1 or word.isdigit():
                spell_buf += word
            else:
                if spell_buf:
                    processed.append(spell_buf.capitalize())
                    spell_buf = ""
                processed.append(self.vocab_mapping.get(word, word))

        if spell_buf:
            processed.append(spell_buf.capitalize())

        return " ".join(processed)

    def _translate_ollama(self, raw: str) -> str:
        url = "http://localhost:11434/api/generate"
        prompt = f"Hãy đóng vai chuyên gia ngôn ngữ ký hiệu. Dịch từ khóa sau thành một câu tiếng Việt hoàn chỉnh, tự nhiên. Chỉ trả kết quả, không giải thích. Từ khóa: '{raw}'"
        try:
            res = requests.post(url, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=10)
            if res.status_code == 200:
                return res.json().get("response", "").strip()
        except Exception as e:
            print(f"[Ollama Error] Không thể kết nối. Vui lòng bật phần mềm Ollama. Lỗi: {e}")
        return None

    def _local_grammar_correction(self, raw_text: str) -> str:
        text = raw_text.lower().strip()
        
        # 1. Các mẫu câu khớp chính xác (ưu tiên cao nhất)
        templates = {
            "tôi tên phat": "Mình tên là phat.",
            "xin chào mọi người": "Xin chào tất cả mọi người!",
            "hôm nay tôi trình bày phát hiện ngôn ngữ ký hiệu": "Hôm nay tôi sẽ trình bày về nhận diện ngôn ngữ ký hiệu.",
            "bạn khỏe": "Bạn dạo này có khỏe không?",
            "cảm ơn bạn": "Cảm ơn bạn rất nhiều.",
            "tôi đi dạo": "Tôi đang đi dạo.",
            "chúc mừng ngày phụ nữ việt nam 20/10": "Chúc mừng ngày Phụ nữ Việt Nam 20/10!",
            "cảm ơn mọi người lắng nghe": "Cảm ơn mọi người đã lắng nghe.",
            "bạn tên là gì" : "Tên của bạn là gì?",
            "tôi sinh sống TP. Hồ Chí Minh" : "Tôi sống tại TP. Hồ Chí Minh.",
        }
        
        if text in templates:
            return templates[text]
            
        # 2. Xử lý bằng luật Ngữ pháp (Rule-based NLP qua Regex)
        # Thêm dấu phẩy sau các trạng từ chỉ thời gian ở đầu câu
        text = re.sub(r'^(hôm nay|ngày mai|hôm qua|buổi sáng|buổi trưa|buổi tối|buổi đêm)\s+(.*)', r'\1, \2', text)
        
        # Thêm từ "rất" trước các tính từ cảm xúc/trạng thái để câu tự nhiên hơn
        text = re.sub(r'\b(vui|buồn|tức giận|mệt|khỏe)\b', r'rất \1', text)
        text = text.replace("rất rất", "rất") # Chống lặp từ
        
        # Phát hiện câu hỏi
        is_question = any(qw in text for qw in ["không?", "vì sao?", "bao giờ?", "khi nào", "là gì", "thế nào?"])
            
        # 3. Chuẩn hóa văn bản cuối cùng
        if text:
            text = re.sub(r'\s+', ' ', text).strip() # Xóa khoảng trắng thừa
            text = text[0].upper() + text[1:]
            
            if is_question and not text.endswith('?'):
                text = re.sub(r'[.!]$', '', text)
                text += '?'
            elif not is_question and not text.endswith(('.', '!', '?')):
                text += '.'
        return text

    def translate_text(self, raw: str) -> str:
        print(f"[Gloss Raw] {raw}")

        if USE_GEMINI and gemini_model:
            try:
                prompt = f"Hãy đóng vai một chuyên gia ngôn ngữ ký hiệu. Dịch các từ khóa sau thành một câu tiếng Việt hoàn chỉnh, tự nhiên và đúng ngữ pháp. Chỉ trả về kết quả câu dịch, không giải thích gì thêm. Từ khóa: '{raw}'"
                response = gemini_model.generate_content(prompt)
                final_text = response.text.strip()
                print(f"[Gemini Translator] {final_text}")
                return final_text
            except Exception as e:
                error_str = str(e).lower()
                if "quota" in error_str or "retry" in error_str or "429" in error_str:
                    print("\n[GEMINI API LIMIT] CẢNH BÁO: Bạn đã chạm giới hạn dịch miễn phí của Google (15 câu/phút). Vui lòng đợi khoảng 1 phút rồi mới dịch tiếp!\n")
                else:
                    print(f"[GEMINI API ERROR] {e}")
                return self._local_grammar_correction(raw)
                
        elif USE_LOCAL_OLLAMA:
            final_text = self._translate_ollama(raw)
            if final_text:
                print(f"[Ollama Translator] {final_text}")
                return final_text
                
        else:
            final_text = self._local_grammar_correction(raw)
            print(f"[Offline Rule Translator] {final_text}")
            return final_text

    def finalize_sentence(self) -> str:
        if not self.buffer:
            return ""
        raw = self._build_raw_sentence()
        self.buffer.clear()
        return self.translate_text(raw)

    def clear(self):
        self.buffer.clear()

    def get_buffer(self) -> list:
        return self.buffer.copy()

    def buffer_length(self) -> int:
        return len(self.buffer)