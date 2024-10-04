import gradio as gr
import copy
import re
from threading import Thread
from transformers import TextIteratorStreamer
from qwen_vl_utils import process_vision_info


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = "<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def _remove_image_special(text):
    text = text.replace("<ref>", "").replace("</ref>", "")
    return re.sub(r"<box>.*?(</box>|$)", "", text)


def is_video_file(filename):
    video_extensions = [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".mpeg"]
    return any(filename.lower().endswith(ext) for ext in video_extensions)


def transform_messages(original_messages):
    transformed_messages = []
    for message in original_messages:
        new_content = []
        for item in message["content"]:
            if "image" in item:
                new_item = {"type": "image", "image": item["image"]}
            elif "text" in item:
                new_item = {"type": "text", "text": item["text"]}
            elif "video" in item:
                new_item = {"type": "video", "video": item["video"]}
            else:
                continue
            new_content.append(new_item)

        new_message = {"role": message["role"], "content": new_content}
        transformed_messages.append(new_message)

    return transformed_messages


def make_vl_demo(model, processor):
    def call_local_model(model, processor, messages):
        messages = transform_messages(messages)

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)

        tokenizer = processor.tokenizer
        streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = {"max_new_tokens": 512, "streamer": streamer, **inputs}

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield generated_text

    def create_predict_fn():
        def predict(_chatbot, task_history):
            chat_query = _chatbot[-1][0]
            query = task_history[-1][0]
            if len(chat_query) == 0:
                _chatbot.pop()
                task_history.pop()
                return _chatbot
            print("User: " + _parse_text(query))
            history_cp = copy.deepcopy(task_history)
            full_response = ""
            messages = []
            content = []
            for q, a in history_cp:
                if isinstance(q, (tuple, list)):
                    if is_video_file(q[0]):
                        content.append({"video": f"file://{q[0]}"})
                    else:
                        content.append({"image": f"file://{q[0]}"})
                else:
                    content.append({"text": q})
                    messages.append({"role": "user", "content": content})
                    messages.append({"role": "assistant", "content": [{"text": a}]})
                    content = []
            messages.pop()

            for response in call_local_model(model, processor, messages):
                _chatbot[-1] = (_parse_text(chat_query), _remove_image_special(_parse_text(response)))

                yield _chatbot
                full_response = _parse_text(response)

            task_history[-1] = (query, full_response)
            print("Qwen-VL-Chat: " + _parse_text(full_response))
            yield _chatbot

        return predict

    def create_regenerate_fn():
        def regenerate(_chatbot, task_history):
            if not task_history:
                return _chatbot
            item = task_history[-1]
            if item[1] is None:
                return _chatbot
            task_history[-1] = (item[0], None)
            chatbot_item = _chatbot.pop(-1)
            if chatbot_item[0] is None:
                _chatbot[-1] = (_chatbot[-1][0], None)
            else:
                _chatbot.append((chatbot_item[0], None))
            _chatbot_gen = predict(_chatbot, task_history)
            for _chatbot in _chatbot_gen:
                yield _chatbot

        return regenerate

    predict = create_predict_fn()
    regenerate = create_regenerate_fn()

    def add_text(history, task_history, text):
        task_text = text
        history = history if history is not None else []
        task_history = task_history if task_history is not None else []
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [(task_text, None)]
        return history, task_history, ""

    def add_file(history, task_history, file):
        history = history if history is not None else []
        task_history = task_history if task_history is not None else []
        history = history + [((file.name,), None)]
        task_history = task_history + [((file.name,), None)]
        return history, task_history

    def reset_user_input():
        return gr.update(value="")

    def reset_state(task_history):
        task_history.clear()
        return []

    with gr.Blocks() as demo:
        gr.Markdown("""<center><font size=8>Qwen2-VL OpenVINO demo</center>""")

        chatbot = gr.Chatbot(label="Qwen2-VL", elem_classes="control-height", height=500)
        query = gr.Textbox(lines=2, label="Input")
        task_history = gr.State([])

        with gr.Row():
            addfile_btn = gr.UploadButton("📁 Upload (上传文件)", file_types=["image", "video"])
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")
            empty_bin = gr.Button("🧹 Clear History (清除历史)")

        submit_btn.click(add_text, [chatbot, task_history, query], [chatbot, task_history]).then(
            predict, [chatbot, task_history], [chatbot], show_progress=True
        )
        submit_btn.click(reset_user_input, [], [query])
        empty_bin.click(reset_state, [task_history], [chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)
        addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True)

        gr.Markdown(
            """\
<font size=2>Note: This demo is governed by the original license of Qwen2-VL. \
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
including hate speech, violence, pornography, deception, etc. \
(注：本演示受Qwen2-VL的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)"""
        )

    return demo


import gradio as gr
import modelscope_studio as mgr
from transformers import TextIteratorStreamer
import librosa
from threading import Thread


def make_audio_demo(model, processor):
    def add_text(chatbot, task_history, input):
        text_content = input.text
        content = []
        if len(input.files) > 0:
            for i in input.files:
                content.append({"audio": i.path})
        if text_content:
            content.append({"text": text_content})
        task_history.append({"role": "user", "content": content})

        chatbot.append(
            [
                {
                    "text": input.text,
                    "files": input.files,
                },
                None,
            ]
        )
        return chatbot, task_history, None

    def add_file(chatbot, task_history, audio_file):
        """Add audio file to the chat history."""
        task_history.append({"role": "user", "content": [{"audio": audio_file.name}]})
        chatbot.append((f"[Audio file: {audio_file.name}]", None))
        return chatbot, task_history

    def reset_user_input():
        """Reset the user input field."""
        return gr.Textbox.update(value="")

    def reset_state():
        """Reset the chat history."""
        return [], []

    def regenerate(chatbot, task_history):
        """Regenerate the last bot response."""
        if task_history and task_history[-1]["role"] == "assistant":
            task_history.pop()
            chatbot.pop()
        if task_history:
            yield predict(chatbot, task_history)
        return chatbot, task_history

    def predict(chatbot, task_history):
        """Generate a response from the model."""

        audios = []
        for message in task_history:
            if isinstance(message["content"], list):
                print(message)
                for ele in message["content"]:
                    if ele.get("audio") is not None:
                        audios.append(librosa.load(ele["audio"], sr=processor.feature_extractor.sampling_rate)[0])
        text = processor.apply_chat_template(
            [{"role": "system", "content": [{"text": "You are a helpful assistant."}]}] + task_history, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
        streamer = TextIteratorStreamer(processor.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {"max_new_tokens": 512, "streamer": streamer, **inputs}
        chatbot.append([None, ""])
        task_history.append({"role": "assistant", "content": [{"text": ""}]})
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            chatbot[-1][-1] = generated_text
            task_history[-1]["content"][0]["text"] = generated_text
            yield chatbot, task_history

    with gr.Blocks() as demo:
        gr.Markdown("""<center><font size=8>OpenVINO Qwen2-Audio-Instruct Bot</center>""")
        chatbot = mgr.Chatbot(label="Qwen2-Audio-7B-Instruct", elem_classes="control-height", height=750)
        user_input = mgr.MultimodalInput(
            interactive=True,
            sources=["microphone", "upload"],
            submit_button_props=dict(value="🚀 Submit (发送)"),
            upload_button_props=dict(value="📁 Upload (上传文件)", show_progress=True),
        )
        task_history = gr.State([])

        with gr.Row():
            empty_bin = gr.Button("🧹 Clear History (清除历史)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")
        user_input.submit(fn=add_text, inputs=[chatbot, task_history, user_input], outputs=[chatbot, task_history, user_input], concurrency_limit=40).then(
            predict, [chatbot, task_history], [chatbot, task_history], show_progress=True
        )
        empty_bin.click(reset_state, outputs=[chatbot, task_history], show_progress=True, concurrency_limit=40)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot, task_history], show_progress=True, concurrency_limit=40)

    return demo
