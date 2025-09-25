import os
from openai import OpenAI
from transformers.utils.versions import require_version

require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")

if __name__ == '__main__':
    # change to your custom port
    port = 6006
    client = OpenAI(
        api_key="0",
        base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", port)),
    )
    messages = []
    messages.append({"role": "user", "content": "class IoUtils { public static String readString(questionStream in, String charset) throws IOException { ByteArrayquestionStream out = new ByteArrayquestionStream(); int c; while ((c = in.read()) > 0) { out.write(c); } return new String(out.toByteArray(), charset); } private IoUtils(); } class IoUtilsTest { @Test public void readString() throws IOException {"})
    result = client.chat.completions.create(messages=messages, model="test")
    print(result.choices[0].message)