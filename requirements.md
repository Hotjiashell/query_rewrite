现在我要持续改进query改写流程，你帮我把baseline、评估的框架搭起来吧。我的输入数据是如 @data/dialog_example.json 那样的对话数据。我需要调用大模型根据对话数据生成一个query，再用这个query调用 @search.py召回案例，看看在top-1,3,5,10的召回率。要求能配置大模型的url mode_name api-key，默认关闭思考模式，关闭的方式是：

chat_response = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
    }, 
)
要求保留生成的query和检索轨迹（案例的话只留id，tittle就够了），要求可指定并行数

baseline的方法就是直接调用一次 @prompt.py里的BASELINE_PROMPT，改进后方案先不管，但是你要给我一个良好的设计，比如gen_query接口，然后我只需要实现这个接口？
返回的格式你可以看 @search.py，注意他这个api有点问题，topN的N不固定，可能到10，可能到7，可能到5，可能到......，你先按顺序把所有top找出来，不要用topN那个N。