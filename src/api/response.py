import os
from sage.src.api import Gemini, GPT, QwenVL, Molmo2, Qwen3_VL
from typing import Optional, Union, List
from sage.src.functions.utils.extract import extract_frames
from sage.utils.json_parser import clean_json
from sage.src.functions.utils.temporal import get_video_duration


def get_response(
    message,
    sys_prompt=None,
    model_name="gemini:gemini-2.5-flash",
    media_urls=None,
    media_type=None,
    history=[],
    max_new_tokens=None,
    model: Optional[Union[QwenVL, Molmo2]] = None,
    return_ids=False,
    temperature=None,
    tool_call_clients: List[object] = None,
    **kwargs,
):
    completion_ids = None
    prompt_ids = None
    if len(history) == 0 and sys_prompt is not None and len(sys_prompt) > 0:
        history.append({"role": "system", "content": sys_prompt})
    if "gemini" in model_name:
        gemini_client = Gemini()
        if sys_prompt is not None and len(sys_prompt) > 0:
            message = sys_prompt + "\n" + message
        response = gemini_client.get_response(
            query=message,
            media_paths=media_urls,
            media_type=media_type,
            model_name=model_name,
            temperature=temperature,
        )["answer"]
        messages = None
    elif "gpt" in model_name:
        gpt_client = GPT()
        if media_type == "video":
            video_path = media_urls[0]
            video_duration = get_video_duration(video_path)
            if video_duration > 0:
                gpt_max_frames = int(os.environ.get("GPT_MAX_FRAMES", "16"))
                frames = extract_frames(video_path, 0, video_duration, num_frames=min(gpt_max_frames, max(2, int(video_duration))))
                media_urls = frames
            else:
                media_urls = []
        if ":" in model_name:
            gpt_model = model_name.split(":")[-1]
        elif "/" in model_name:
            gpt_model = model_name.split("/")[-1]
        else:
            gpt_model = model_name
        response, messages = gpt_client.get_response(
            prompt=message, history=history, image_urls=media_urls, model=gpt_model, temperature=temperature
        )
    elif "qwen" in model_name or "molmo2" in model_name:
        try:
            response = model.get_response(
                    prompt=message, media=media_urls, media_type=media_type, system_prompt=sys_prompt, 
                    max_new_tokens=max_new_tokens, return_ids=return_ids, 
                    temperature=temperature, **kwargs
                )
            if return_ids:
                response, completion_ids, prompt_ids, attention_mask = response
        except Exception as e:
            print("Error in response in API call: ", e, "for message of length: ", len(message), "with media_urls: ", media_urls, "and media_type: ", media_type)
            response = None
            completion_ids = []
            prompt_ids = []
            attention_mask = []
        messages = None

    if response is None:
        if return_ids:
            return None, messages, [], [], []
        return None, messages
    
    if "<json>" in response:
        try:
            while "<json>" in response:
                response = response.split("<json>")[1].strip().split("</json>")[0].strip()
            response = clean_json(response)
        except Exception as e:
            print("Error parsing <json> response: ", e)
    elif "```json" in response:
        try:
            response = response.split("```json")[1].split("```")[0]
            response = clean_json(response)
        except Exception as e:
            print("Error parsing ```json response: ", e)
    else:
        # Bare JSON with no <json> tags or ```json fences (e.g. GPT reasoner
        # responses). Without this, the raw JSON string is returned and the
        # caller stores the whole blob instead of extracting final_answer.
        stripped = response.strip() if isinstance(response, str) else response
        if isinstance(stripped, str) and stripped.startswith("{"):
            try:
                response = clean_json(stripped)
            except Exception as e:
                print("Error parsing bare JSON response: ", e)

    if return_ids:
        return response, messages, completion_ids, prompt_ids, attention_mask
    return response, messages
