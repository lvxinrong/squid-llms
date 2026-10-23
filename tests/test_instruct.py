from squid.data.instruct import InstructDataset
from squid.tokenizer import TiktokenTokenizer


def _fake_data():
     return [
         {"instruction": "Convert to passive.", "input": "The chef cooks the meal.",
          "output": "The meal is cooked by the chef."},
         {"instruction": "What is 1+1?", "input": "", "output": "2"},
     ]


def test_instruct_dataset():
     tok = TiktokenTokenizer("gpt2")
     ds = InstructDataset(_fake_data(), tok)

     assert len(ds) == 2

     prompt_ids, response_ids = ds[0]
     prompt_text = tok.decode(prompt_ids)
     response_text = tok.decode(response_ids)

     print(prompt_text)
     print(response_text)

     # prompt 以 ### Response: 收尾，response 就是 output 原文
     assert prompt_text.rstrip().endswith("### Response:")
     assert response_text == "The meal is cooked by the chef."

     # input 为空的分支：第二条不该出现 ### Input:
     prompt_ids2, _ = ds[1]
     assert "### Input:" not in tok.decode(prompt_ids2)

