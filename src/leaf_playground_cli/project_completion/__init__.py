import json
import re
import os
from typing import Literal

from leaf_playground.ai_backend.openai import OpenAIBackend, OpenAIBackendConfig
from pydantic import Field


class GeneratorConfig(OpenAIBackendConfig):
    model: Literal["gpt-4", "gpt-4-1106-preview"] = Field(default="gpt-4-1106-preview")


class Generator(OpenAIBackend):
    config_cls = GeneratorConfig
    config: config_cls

    def __init__(self, config: GeneratorConfig, project_dir: str):
        if not os.path.exists(os.path.join(project_dir, ".leaf")):
            raise NotADirectoryError(f"{project_dir} not a leaf project.")
        proj_config = json.load(
            open(os.path.join(project_dir, ".leaf", "project_config.json"), "r", encoding="utf-8")
        )

        super().__init__(config)

        prompt_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "prompts")

        self.package_dir = os.path.abspath(os.path.join(project_dir, proj_config["name"]))
        self.proj_config = proj_config
        self.prompts = {
            "system": self._read_file(os.path.join(prompt_dir, "system.txt")),
            "agents": self._read_file(os.path.join(prompt_dir, "agents.txt")),
            "scene": self._read_file(os.path.join(prompt_dir, "scene.txt"))
        }

    @staticmethod
    def _read_file(file: str):
        with open(file, "r", encoding="utf-8") as f:
            prompt = f.read()
        return prompt

    @staticmethod
    def _extract_code(file_name: str, code: str):
        pattern = r'<code file="' + file_name + r'.py">(.*?)</code>'
        match = re.search(pattern, code, re.DOTALL)

        if match:
            return match.group(1).strip(), True
        else:
            return code, False

    def run(self):
        try:
            scene_definition_code = self._read_file(os.path.join(self.package_dir, "scene_definition.py"))
        except:
            raise FileNotFoundError(
                "scene_definition.py not found, you must complete this module before using this function."
            )

        messages = [
            {
                "role": "system",
                "content": self.prompts["system"]
            },
            {
                "role": "assistant",
                "content": "我已阅读并完全理解以上文档，并将在接下来完全遵循用户的指示。"
            },
            {
                "role": "user",
                "content": f"以下是 scene_definition.py 代码\n<code file=\"scene_definition.py\">\n"
                           f"{scene_definition_code}\n</code>\n请回复“收到”以接收该代码，不要说其他的话。"
            },
            {
                "role": "assistant",
                "content": "收到"
            },
            {
                "role": "user",
                "content": self.prompts["agents"]
            },
            {
                "role": "assistant",
                "content": "明白"
            }
        ]

        # gen agent modules
        for role_def in self.proj_config["metadata"]["scene_metadata"]["scene_definition"]["roles"]:
            role_name = role_def["name"]
            is_static_role = role_def["is_static"]
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "{{ role_name }} = " + role_name + (
                            ", 这是一个静态角色，你需要结合 scene_definition.py 中的信息和你对业务的理解，"
                            "给出实际的类初始化和动作方法的实现逻辑，而不是框架性代码"
                            if is_static_role else ""
                        )
                    )
                }
            )
            stream = self.client.chat.completions.create(
                messages=messages,
                model=self.config.model,
                max_tokens=4096,
                temperature=0.5,
                stream=True
            )
            response = ""
            for item in stream:
                if item.choices[0].delta.content is not None:
                    content = item.choices[0].delta.content
                    print(content, end="")
                    response += content
            print()

            code, extract_success = self._extract_code(role_name, response)
            with open(os.path.join(self.package_dir, "agents", f"{role_name}.py"), "w", encoding="utf-8") as f:
                f.write(code)
            if extract_success:
                code = f'<code file="{role_name}.py">\n' + code + "\n<\code>"
            messages.append({"role": "assistant", "content": code})

        # gen scene module
        messages.append({"role": "user", "content": self.prompts["scene"]})
        stream = self.client.chat.completions.create(
            messages=messages,
            model=self.config.model,
            max_tokens=4096,
            temperature=0.5,
            stream=True
        )
        response = ""
        for item in stream:
            if item.choices[0].delta.content is not None:
                content = item.choices[0].delta.content
                print(content, end="")
                response += content
        print()

        code, extract_success = self._extract_code("scene", response)
        with open(os.path.join(self.package_dir, "scene.py"), "w", encoding="utf-8") as f:
            f.write(code)


if __name__ == "__main__":
    generator = Generator(config=GeneratorConfig(), project_dir="D:\Repositories\leaf-playground-next\src\dev\mmlu")
    generator.run()
