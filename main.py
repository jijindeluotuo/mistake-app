import os, re, json, uuid, base64, hashlib, time, asyncio, subprocess, sys, tempfile
from typing import List, Dict, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import httpx

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MATH_MODEL = os.getenv("QWEN_MATH_MODEL", "qwen-math-turbo")
QWEN_GENERAL_MODEL = os.getenv("QWEN_GENERAL_MODEL", "qwen-plus")
YOUDAO_APP_KEY = os.getenv("YOUDAO_APP_KEY", "")
YOUDAO_APP_SECRET = os.getenv("YOUDAO_APP_SECRET", "")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))
SYMPY_TIMEOUT = int(os.getenv("SYMPY_TIMEOUT", "15"))

_ds = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL) if DEEPSEEK_API_KEY else None
_qw = AsyncOpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL) if DASHSCOPE_API_KEY else None
SUBJECT_CN = {"math": "数学", "physics": "物理", "chemistry": "化学"}

SANDBOX_RUNNER = r'''
import sys, ast, builtins
ALLOWED_ROOTS = ("sympy", "math", "fractions", "decimal")
def check(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name.split(".")[0] not in ALLOWED_ROOTS:
                    print(f"FORBIDDEN_IMPORT:{n.name}"); sys.exit(2)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_ROOTS:
                print(f"FORBIDDEN_IMPORT:{node.module}"); sys.exit(2)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("eval", "exec", "open", "__import__", "compile"):
                print(f"FORBIDDEN_CALL:{node.func.id}"); sys.exit(2)
code = open(sys.argv[1], encoding="utf-8").read()
check(code)
real_import = builtins.__import__
def safe_import(name, *args, **kwargs):
    if name.split(".")[0] not in ALLOWED_ROOTS:
        raise ImportError(f"import {name} not allowed")
    return real_import(name, *args, **kwargs)
safe_builtins = {
    "print": print, "len": len, "range": range, "int": int, "float": float,
    "str": str, "list": list, "dict": dict, "set": set, "tuple": tuple,
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "pow": pow, "enumerate": enumerate, "zip": zip, "bool": bool,
    "True": True, "False": False, "None": None,
    "__import__": safe_import,
}
try:
    exec(compile(code, "<llm>", "exec"), {"__builtins__": safe_builtins})
except Exception as e:
    print(f"ERROR:{type(e).__name__}:{e}")
    sys.exit(3)
'''

def run_sympy(code: str) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code); code_path = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(SANDBOX_RUNNER); runner_path = f.name
    try:
        proc = subprocess.run([sys.executable, runner_path, code_path],
                              capture_output=True, text=True, timeout=SYMPY_TIMEOUT)
        out = proc.stdout.strip()
        result = None
        for line in out.splitlines():
            if line.startswith("RESULT:"):
                result = line[len("RESULT:"):].strip()
        if proc.returncode != 0:
            return {"ok": False, "result": None, "error": (proc.stderr or out)[-300:]}
        if result is None:
            return {"ok": False, "result": None, "error": "no RESULT line"}
        return {"ok": True, "result": result, "error": None}
    except subprocess.TimeoutExpired:
        return {"ok": False, "result": None, "error": "timeout"}
    finally:
        for p in (code_path, runner_path):
            try: os.unlink(p)
            except OSError: pass

async def deepseek_json(system: str, user: str, temperature: float = 0.2) -> dict:
    if not _ds: raise RuntimeError("DEEPSEEK_API_KEY 未配置")
    resp = await _ds.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=temperature, timeout=REQUEST_TIMEOUT,
    )
    return json.loads(resp.choices[0].message.content)

async def qwen_solve(system: str, user: str, subject: str) -> str:
    if not _qw: return ""
    model = QWEN_MATH_MODEL if subject == "math" else QWEN_GENERAL_MODEL
    resp = await _qw.chat.completions.create(
        model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1, timeout=REQUEST_TIMEOUT,
    )
    return resp.choices[0].message.content or ""

def normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\\left|\\right|\\,|\\;|\\!|\\ ", "", s)
    s = re.sub(r"\s+", "", s)
    return s.replace("−", "-").replace("×", "*").replace("÷", "/").replace("，", ",")

async def answers_equivalent(a: str, b: str, subject: str) -> bool:
    if normalize(a) == normalize(b): return True
    prompt = (f"科目：{SUBJECT_CN.get(subject, subject)}。判断以下两个答案是否等价。"
              f'只输出 JSON：{{"equivalent": true/false}}。\n答案A：{a}\n答案B：{b}')
    try:
        data = await deepseek_json("你是严谨的阅卷老师。", prompt, temperature=0.0)
        return bool(data.get("equivalent"))
    except Exception:
        return False

async def split_knowledge_points(subject: str, stem: str, tags: List[str]) -> List[str]:
    if tags: return tags
    prompt = (f"科目：{SUBJECT_CN.get(subject, subject)}。\n题目：{stem}\n"
              "请输出该题涉及的 3-5 个具体考点，不要模糊表述。"
              '只输出 JSON：{"tags": ["考点1", "考点2"]}')
    data = await deepseek_json("你是高考命题研究员，擅长拆解考点。", prompt, temperature=0.2)
    return data.get("tags") or ["未分类考点"]

async def generate_questions(subject: str, stem: str, tags: List[str], count: int) -> List[dict]:
    prompt = (f"科目：{SUBJECT_CN.get(subject, subject)}。\n原题：{stem}\n"
              f"考点：{', '.join(tags)}\n"
              f"请生成 {count} 道同考点变式题，覆盖不同考法，不要重复原题数字。\n"
              "每道题必须可求解、答案唯一、难度贴近高考。\n"
              '只输出 JSON：{"questions":[{"stem":"题干","answer":"最终答案",'
              '"solution":"解析","knowledge_tag":"对应考点","question_type":"choice|fill|solve"}]}')
    data = await deepseek_json("你是高考命题老师，只出可验证的题。", prompt, temperature=0.7)
    return data.get("questions", [])

async def sympy_verify(stem: str, claimed: str) -> Tuple[bool, str]:
    prompt = (f"题目：{stem}\n声称答案：{claimed}\n"
              "请写一段 Python 代码，使用 sympy 从头独立计算正确答案，"
              "不要读取或使用声称答案。最后一行必须是 print('RESULT: ' + str(答案))。"
              '只输出 JSON：{"code": "完整可运行的 Python 代码"}')
    try:
        data = await deepseek_json("你是严谨的数学验证工程师。", prompt, temperature=0.0)
        code = data.get("code", "")
    except Exception:
        return True, "sympy_gen_failed_pass"
    if not code: return True, "sympy_gen_empty_pass"
    r = await asyncio.to_thread(run_sympy, code)
    if not r["ok"]: return True, f"not_computable({r['error'][:30]})"
    if await answers_equivalent(r["result"], claimed, "math"): return True, ""
    return False, "sympy_mismatch"

async def qwen_verify(stem: str, claimed: str, subject: str) -> Tuple[bool, str]:
    if not _qw: return True, "qwen_skipped"
    prompt = f"题目：{stem}\n请独立求解，不要假设任何给定答案。最后一行输出 FINAL: <最终答案>。"
    try:
        text = await qwen_solve("你是严谨的高考阅卷老师。", prompt, subject)
    except Exception:
        return True, "qwen_call_failed_pass"
    m = re.search(r"FINAL:\s*(.+)", text, re.S)
    if not m: return True, "qwen_no_final_pass"
    final = m.group(1).strip().splitlines()[0]
    if await answers_equivalent(final, claimed, subject): return True, ""
    return False, "qwen_mismatch"

async def solution_check(stem: str, answer: str, solution: str) -> Tuple[bool, str]:
    prompt = (f"题目：{stem}\n答案：{answer}\n解析：{solution}\n"
              '判断解析步骤是否与答案一致、是否有明显错误。'
              '只输出 JSON：{"ok": true/false, "reason": "简短原因"}')
    try:
        data = await deepseek_json("你是严谨的教研员。", prompt, temperature=0.0)
        return (True, "") if data.get("ok") else (False, "solution_invalid")
    except Exception:
        return True, "solution_check_failed_pass"

async def verify_one(q: dict, subject: str) -> Tuple[bool, str]:
    stem, claimed = q.get("stem", ""), q.get("answer", "")
    if not stem or not claimed: return False, "missing_stem_or_answer"
    if subject == "math":
        ok, r = await sympy_verify(stem, claimed)
        if not ok: return False, r
    ok, r = await qwen_verify(stem, claimed, subject)
    if not ok: return False, r
    ok, r = await solution_check(stem, claimed, q.get("solution", ""))
    if not ok: return False, r
    return True, ""

class GenerateRequest(BaseModel):
    mistake_id: str
    subject: str = "math"
    stem_latex: str = ""
    knowledge_tags: List[str] = []
    count: int = Field(default=3, ge=1, le=5)

class GeneratedQuestion(BaseModel):
    id: str
    stem_latex: str
    answer_latex: str
    solution_latex: str
    knowledge_tag: str

class GenerateResponse(BaseModel):
    questions: List[GeneratedQuestion]
    discarded_count: int = 0
    discard_reasons: Dict[str, int] = {}

app = FastAPI(title="Mistake Verify Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"ok": True, "deepseek": bool(_ds), "qwen": bool(_qw), "ocr": bool(YOUDAO_APP_KEY)}

@app.post("/generate-and-verify", response_model=GenerateResponse)
async def generate_and_verify(req: GenerateRequest):
    if not _ds:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY 未配置")
    try:
        tags = await split_knowledge_points(req.subject, req.stem_latex, req.knowledge_tags)
        raw = await generate_questions(req.subject, req.stem_latex, tags, req.count)
        passed: List[GeneratedQuestion] = []
        reasons: Dict[str, int] = {}
        for q in raw:
            ok, reason = await verify_one(q, req.subject)
            if ok:
                passed.append(GeneratedQuestion(
                    id=str(uuid.uuid4()), stem_latex=q.get("stem", ""),
                    answer_latex=q.get("answer", ""), solution_latex=q.get("solution", ""),
                    knowledge_tag=q.get("knowledge_tag", tags[0] if tags else "")))
            else:
                reasons[reason] = reasons.get(reason, 0) + 1
        return GenerateResponse(questions=passed, discarded_count=len(raw) - len(passed),
                                discard_reasons=reasons)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"pipeline_failed: {type(e).__name__}: {e}")

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    if not YOUDAO_APP_KEY or not YOUDAO_APP_SECRET:
        raise HTTPException(status_code=503, detail="OCR 未配置，请先用打字录入")
    try:
        content = await file.read()
        salt, curtime = str(uuid.uuid4()), str(int(time.time()))
        sign = hashlib.sha256((YOUDAO_APP_KEY + salt + curtime + YOUDAO_APP_SECRET).encode()).hexdigest()
        data = {"img": base64.b64encode(content).decode(), "langType": "auto",
                "detectType": "10011", "imageType": "1", "appKey": YOUDAO_APP_KEY,
                "salt": salt, "curtime": curtime, "sign": sign, "signType": "v3"}
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post("https://openapi.youdao.com/ocr_formula", data=data)
        payload = resp.json()
        if str(payload.get("errorCode")) != "0":
            raise RuntimeError(f"youdao_error: {payload.get('errorCode')} {payload.get('msg')}")
        lines = []
        for region in (payload.get("Result") or {}).get("regions", []):
            for line in region.get("lines", []):
                t = (line.get("text") or "").strip()
                if t: lines.append(t)
        return {"text": "\n".join(lines)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ocr_failed: {e}")