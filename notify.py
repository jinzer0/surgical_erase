import os
import time
import traceback
import socket
from functools import wraps
from notifiers import get_notifier
from dotenv import load_dotenv
import sys

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")



def get_notified(token=API_TOKEN, chat_id=CHAT_ID, task_name="Deep Learning Task"):
    """
    학습/추론의 시작, 성공, 실패(크래시)를 텔레그램으로 알리는 데코레이터

    Example:
    --------
    from notify import get_notified
    TOKEN = "YOUR_BOT_TOKEN"
    CHAT_ID = "YOUR_CHAT_ID"

    @get_notified(token=TOKEN, chat_id=CHAT_ID, task_name="ResNet50_Training")
    def train_model(epochs, lr, batch_size=32):
        print(f"학습 중... (Epochs: {epochs}, LR: {lr})")

        # 억지로 에러를 발생시켜 테스트해보고 싶다면 아래 주석 해제
        # raise ValueError("GPU Memory Out of Index!")

        time.sleep(3)
        return {"final_loss": 0.04, "accuracy": 0.96}

    if __name__ == "__main__":
        # 인자를 자유롭게 넣어도 데코레이터가 다 받아서 전달합니다.
        train_model(epochs=10, lr=0.001, batch_size=64)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            notifier = get_notifier("telegram")
            host_name = socket.gethostname()  # 어떤 서버에서 돌고 있는지 확인용
            cli_command = " ".join(sys.argv)  # 어떤 커맨드로 실행했는지 확인용
            start_time = time.time()

            # 1. 시작 알림
            start_msg = (
                f"🚀 *[{host_name}]* 작업 시작\n"
                f"━━━━━━━━━━━━━━━\n"
                f"📌 *Task:* `{task_name}`\n"
                f"📂 *Func:* `{func.__name__}`\n"
                f"💻 *Command:* `{cli_command}`"
            )
            res = notifier.notify(token=token, chat_id=chat_id, message=start_msg, parse_mode="markdown")
            data = res.response.json()
            msg_id = data["result"]["message_id"]

            try:
                # 2. 본 함수(학습/추론) 실행
                # 원래 인자들(*args, **kwargs)을 그대로 전달합니다.
                result = func(*args, **kwargs)

                # 3. 성공 완료 알림
                end_time = time.time()
                elapsed = (end_time - start_time) / 60

                success_msg = (
                    f"✅ *[{host_name}] {task_name}* 작업 완료!\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"⏱ *소요 시간:* `{elapsed:.2f}분`\n"
                    f"📊 *결과 요약:*\n```python\n{result}\n```"
                )
                notifier.notify(token=token, chat_id=chat_id, message=success_msg, parse_mode="markdown", reply_to_message_id=msg_id)
                return result

            except Exception as e:
                # 4. 크래시(에러) 발생 시 알림
                error_type = type(e).__name__
                error_traceback = traceback.format_exc()[-3900:] # 에러 로그 마지막 200자

                fail_msg = (
                    f"❌ *[{host_name}] {task_name}* 크래시 발생!\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"⚠️ *Error:* `{type(e).__name__}`\n"
                    f"🔍 *Traceback Summary:*\n"
                    f"```text\n{error_traceback}\n```"
                )
                notifier.notify(token=token, chat_id=chat_id, message=fail_msg, parse_mode="markdown", reply_to_message_id=msg_id)

                # 에러를 다시 발생시켜 시스템이 에러를 인지하게 함
                raise e

        return wrapper
    return decorator
