SIMCLS_DATA_PATH = "src/consum/utils/simcls_test_data.json"
SIMCLS_MODEL_PATH = "src/consum/utils/simcls_test_model/scorer.bin"

HYPOTHESES = [
    "Woman gives her kidney away in big \"super swap\" Charity is the idea that strangers can be given kidneys back to someone else. Blood samples of donors and recipients are processed to reveal match, hospital says. Chain of surgeries to be wrapped up Friday. In March, hospital will hold banquet to thank donors, recipients and support personnel.",
    "Zully Broussard donated her kidney to a stranger after receiving a donation from another person. Her generous act set off a chain of kidney transplants in San Francisco. A computer programmer who received a kidney transplant paid itforward by developing MatchGrid. That matched up donors with matching relatives using genetic data.",
    "Woman's donation is matched with genetic profiles from a donor and recipient. That sends a chain reaction, like dominoes falling. The chain of surgeries is to be wrapped up Friday. Doctors are extracting six kidneys and inserting them into six recipients.  Such long-chain transplanting is rare.",
    "Six people will receive kidney transplants after woman gives one of her to a stranger. Five surgeons and other staff perform the surgeries. The chain of surgeries is to be wrapped up Friday. A computer programmer used his gift to build a program that matches donor pairs quickly. A former patient posted a message on Facebook in her name.",
]

REFERENCES = [
    "Zully Broussard decided to give a kidney to a stranger .\nA new computer program helped her donation spur transplants for six kidney patients .",
]* len(HYPOTHESES)

SOURCES = [
    "(CNN)Share, and your gift will be multiplied. That may sound like an esoteric adage, but when Zully Broussard selflessly decided to give one of her kidneys to a stranger, her generosity paired up with big data. It resulted in six patients receiving transplants. That surprised and wowed her. \"I thought I was going to help this one person who I don't know, but the fact that so many people can have a life extension, that's pretty big,\" Broussard told CNN affiliate KGO. She may feel guided in her generosity by a higher power. \"Thanks for all the support and prayers,\" a comment on a Facebook page in her name read. \"I know this entire journey is much bigger than all of us. I also know I'm just the messenger.\" CNN cannot verify the authenticity of the page. But the power that multiplied Broussard's gift was data processing of genetic profiles from donor-recipient pairs. It works on a simple swapping principle but takes it to a much higher level, according to California Pacific Medical Center in San Francisco. So high, that it is taking five surgeons, a covey of physician assistants, nurses and anesthesiologists, and more than 40 support staff to perform surgeries on 12 people. They are extracting six kidneys from donors and implanting them into six recipients. \"The ages of the donors and recipients range from 26 to 70 and include three parent and child pairs, one sibling pair and one brother and sister-in-law pair,\" the medical center said in a statement. The chain of surgeries is to be wrapped up Friday. In late March, the medical center is planning to hold a reception for all 12 patients. Here's how the super swap works, according to California Pacific Medical Center. Say, your brother needs a kidney to save his life, or at least get off of dialysis, and you're willing to give him one of yours. But then it turns out that your kidney is not a match for him, and it's certain his body would reject it. Your brother can then get on a years-long waiting list for a kidney coming from an organ donor who died. Maybe that will work out -- or not, and time could run out for him. Alternatively, you and your brother could look for another recipient-living donor couple like yourselves -- say, two more siblings, where the donor's kidney isn't suited for his sister, the recipient. But maybe your kidney is a match for his sister, and his kidney is a match for your brother. So, you'd do a swap. That's called a paired donation. It's a bit of a surgical square dance, where four people cross over partners temporarily and everybody goes home smiling. But instead of a square dance, Broussard's generous move set off a chain reaction, like dominoes falling. Her kidney, which was removed Thursday, went to a recipient, who was paired with a donor. That donor's kidney went to the next recipient, who was also paired with a donor, and so on. On Friday, the last donor will give a kidney to someone who has been biding time on one of those deceased donor lists to complete the chain. Such long-chain transplanting is rare. It's been done before, California Pacific Medical Center said in a statement, but matching up the people in the chain has been laborious and taken a long time. That changed when a computer programmer named David Jacobs received a kidney transplant. He had been waiting on a deceased donor list, when a live donor came along -- someone nice enough to give away a kidney to a stranger. Jacobs paid it forward with his programming skills, creating MatchGrid, a program that genetically matches up donor pairs or chains quickly. \"When we did a five-way swap a few years ago, which was one of the largest, it took about three to four months. We did this in about three weeks,\" Jacobs said. But this chain wouldn't have worked so quickly without Broussard's generosity -- or may not have worked at all. \"The significance of the altruistic donor is that it opens up possibilities for pairing compatible donors and recipients,\" said Dr. Steven Katznelson. \"Where there had been only three or four options, with the inclusion of the altruistic donor, we had 140 options to consider for matching donors and recipients.\" And that's divine, Broussard's friend Shirley Williams wrote in a comment her on Broussard's Facebook page. \"You are a true angel my friend.\""
]* len(HYPOTHESES)

FENICE_SCORES = [
    {
        "fenice_score": 0.5384499870764557,
        "fenice_alignments": [
            {
                "score": 0.9443128351122141,
                "summary_claim": "Woman gives her kidney away in big'super swap'.",
                "source_passage": "It's a bit of a surgical square dance, where four people cross over partners temporarily and everybody goes home smiling. But instead of a square dance, Broussard's generous move set off a chain reaction, like dominoes falling. Her kidney, which was removed Thursday, went to a recipient, who was paired with a donor. That donor's kidney went to the next recipient, who was also paired with a donor, and so on. On Friday, the last donor will give a kidney to someone who has been biding time on one of those deceased donor lists to complete the chain."
            },
            {
                "score": 0.9983648621710017,
                "summary_claim": "Charity is the idea that strangers can be given kidneys back to someone else.",
                "source_passage": "It's been done before, California Pacific Medical Center said in a statement, but matching up the people in the chain has been laborious and taken a long time. That changed when a computer programmer named David Jacobs received a kidney transplant. He had been waiting on a deceased donor list, when a live donor came along -- someone nice enough to give away a kidney to a stranger. Jacobs paid it forward with his programming skills, creating MatchGrid, a program that genetically matches up donor pairs or chains quickly. \"When we did a five-way swap a few years ago, which was one of the largest, it took about three to four months."
            },
            {
                "score": -0.2194424271583557,
                "summary_claim": "Blood samples of donors and recipients are processed to reveal match, hospital says.",
                "source_passage": "DOCUMENT"
            },
            {
                "score": 0.9982101953646634,
                "summary_claim": "Chain of surgeries to be wrapped up Friday.",
                "source_passage": "The chain of surgeries is to be wrapped up Friday. In late March, the medical center is planning to hold a reception for all 12 patients. Here's how the super swap works, according to California Pacific Medical Center. Say, your brother needs a kidney to save his life, or at least get off of dialysis, and you're willing to give him one of yours. But then it turns out that your kidney is not a match for him, and it's certain his body would reject it."
            },
            {
                "score": -0.02919553010724485,
                "summary_claim": "In March, hospital will hold banquet to thank donors, recipients and support personnel.",
                "source_passage": "In late March, the medical center is planning to hold a reception for all 12 patients."
            }
        ]
    },
    {
        "fenice_score": 0.878517246710544,
        "fenice_alignments": [
            {
                "score": 0.5922185480594635,
                "summary_claim": "Zully Broussard donated her kidney to a stranger after receiving a donation from another person.",
                "source_passage": "DOCUMENT"
            },
            {
                "score": 0.9974471434252337,
                "summary_claim": "Her generous act set off a chain of kidney transplants in San Francisco.",
                "source_passage": "Her kidney, which was removed Thursday, went to a recipient, who was paired with a donor. That donor's kidney went to the next recipient, who was also paired with a donor, and so on. On Friday, the last donor will give a kidney to someone who has been biding time on one of those deceased donor lists to complete the chain. Such long-chain transplanting is rare. It's been done before, California Pacific Medical Center said in a statement, but matching up the people in the chain has been laborious and taken a long time."
            },
            {
                "score": 0.9993044999137055,
                "summary_claim": "A computer programmer who received a kidney transplant paid itforward by developing MatchGrid.",
                "source_passage": "Such long-chain transplanting is rare. It's been done before, California Pacific Medical Center said in a statement, but matching up the people in the chain has been laborious and taken a long time. That changed when a computer programmer named David Jacobs received a kidney transplant. He had been waiting on a deceased donor list, when a live donor came along -- someone nice enough to give away a kidney to a stranger. Jacobs paid it forward with his programming skills, creating MatchGrid, a program that genetically matches up donor pairs or chains quickly."
            },
            {
                "score": 0.9250987954437733,
                "summary_claim": "MatchGrid matched up donors with matching relatives using genetic data.",
                "source_passage": "DOCUMENT"
            }
        ]
    },
    {
        "fenice_score": 0.9963228945474839,
        "fenice_alignments": [
            {
                "score": 0.9860988515429199,
                "summary_claim": "Woman's donation is matched with genetic profiles from a donor and recipient.",
                "source_passage": "\"Thanks for all the support and prayers,\" a comment on a Facebook page in her name read. \"I know this entire journey is much bigger than all of us. I also know I'm just the messenger.\" CNN cannot verify the authenticity of the page. But the power that multiplied Broussard's gift was data processing of genetic profiles from donor-recipient pairs."
            },
            {
                "score": 0.9978507605264895,
                "summary_claim": "That sends a chain reaction, like dominoes falling.",
                "source_passage": "That's called a paired donation. It's a bit of a surgical square dance, where four people cross over partners temporarily and everybody goes home smiling. But instead of a square dance, Broussard's generous move set off a chain reaction, like dominoes falling. Her kidney, which was removed Thursday, went to a recipient, who was paired with a donor. That donor's kidney went to the next recipient, who was also paired with a donor, and so on."
            },
            {
                "score": 0.9992848303518258,
                "summary_claim": "The chain of surgeries is to be wrapped up Friday.",
                "source_passage": "On Friday, the last donor will give a kidney to someone who has been biding time on one of those deceased donor lists to complete the chain. Such long-chain transplanting is rare. It's been done before, California Pacific Medical Center said in a statement, but matching up the people in the chain has been laborious and taken a long time. That changed when a computer programmer named David Jacobs received a kidney transplant. He had been waiting on a deceased donor list, when a live donor came along -- someone nice enough to give away a kidney to a stranger."
            },
            {
                "score": 0.9989732409012504,
                "summary_claim": "Doctors are extracting six kidneys and inserting them into six recipients.",
                "source_passage": "It works on a simple swapping principle but takes it to a much higher level, according to California Pacific Medical Center in San Francisco. So high, that it is taking five surgeons, a covey of physician assistants, nurses and anesthesiologists, and more than 40 support staff to perform surgeries on 12 people. They are extracting six kidneys from donors and implanting them into six recipients. \"The ages of the donors and recipients range from 26 to 70 and include three parent and child pairs, one sibling pair and one brother and sister-in-law pair,\" the medical center said in a statement. The chain of surgeries is to be wrapped up Friday."
            },
            {
                "score": 0.9994067894149339,
                "summary_claim": "Such long-chain transplanting is rare.",
                "source_passage": "Such long-chain transplanting is rare. It's been done before, California Pacific Medical Center said in a statement, but matching up the people in the chain has been laborious and taken a long time. That changed when a computer programmer named David Jacobs received a kidney transplant. He had been waiting on a deceased donor list, when a live donor came along -- someone nice enough to give away a kidney to a stranger. Jacobs paid it forward with his programming skills, creating MatchGrid, a program that genetically matches up donor pairs or chains quickly."
            }
        ]
    },
    {
        "fenice_score": 0.9908737074234523,
        "fenice_alignments": [
            {
                "score": 0.9974625635659322,
                "summary_claim": "Six people will receive kidney transplants after woman gives one of her to a stranger.",
                "source_passage": "(CNN)Share, and your gift will be multiplied. That may sound like an esoteric adage, but when Zully Broussard selflessly decided to give one of her kidneys to a stranger, her generosity paired up with big data. It resulted in six patients receiving transplants. That surprised and wowed her. \"I thought I was going to help this one person who I don't know, but the fact that so many people can have a life extension, that's pretty big,\" Broussard told CNN affiliate KGO."
            },
            {
                "score": 0.9988405763870105,
                "summary_claim": "Five surgeons and other staff perform the surgeries.",
                "source_passage": "But the power that multiplied Broussard's gift was data processing of genetic profiles from donor-recipient pairs. It works on a simple swapping principle but takes it to a much higher level, according to California Pacific Medical Center in San Francisco. So high, that it is taking five surgeons, a covey of physician assistants, nurses and anesthesiologists, and more than 40 support staff to perform surgeries on 12 people. They are extracting six kidneys from donors and implanting them into six recipients. \"The ages of the donors and recipients range from 26 to 70 and include three parent and child pairs, one sibling pair and one brother and sister-in-law pair,\" the medical center said in a statement."
            },
            {
                "score": 0.9992848303518258,
                "summary_claim": "The chain of surgeries is to be wrapped up Friday.",
                "source_passage": "On Friday, the last donor will give a kidney to someone who has been biding time on one of those deceased donor lists to complete the chain. Such long-chain transplanting is rare. It's been done before, California Pacific Medical Center said in a statement, but matching up the people in the chain has been laborious and taken a long time. That changed when a computer programmer named David Jacobs received a kidney transplant. He had been waiting on a deceased donor list, when a live donor came along -- someone nice enough to give away a kidney to a stranger."
            },
            {
                "score": 0.9993018718669191,
                "summary_claim": "A computer programmer used his gift to build a program that matches donor pairs quickly.",
                "source_passage": "That changed when a computer programmer named David Jacobs received a kidney transplant. He had been waiting on a deceased donor list, when a live donor came along -- someone nice enough to give away a kidney to a stranger. Jacobs paid it forward with his programming skills, creating MatchGrid, a program that genetically matches up donor pairs or chains quickly. \"When we did a five-way swap a few years ago, which was one of the largest, it took about three to four months. We did this in about three weeks,\" Jacobs said."
            },
            {
                "score": 0.9594786949455738,
                "summary_claim": "A former patient posted a message on Facebook in her name.",
                "source_passage": "DOCUMENT"
            }
        ]
    }
]

FIZZ_SCORES = [
    {
        "fizz_score": 0.006958381272852421,
        "filtered_atomic_facts": "A woman gave her kidney away. The kidney was part of a big \"super swap\". Charity is the idea that strangers can give kidneys to someone else. Blood samples are processed. Blood samples belong to donors and recipients. The purpose of processing the blood samples is to reveal match. The hospital is involved in the process. Chain of surgeries is happening. Chain of surgeries will be wrapped up Friday. In March, a hospital will hold a banquet. The banquet will be to thank donors. The banquet will also be to thank recipients. The banquet will also be to thank support personnel."
    },
    {
        "fizz_score": 0.034091394394636154,
        "filtered_atomic_facts": "Zully Broussard donated her kidney. Zully Broussard donated her kidney to a stranger. Zully Broussard received a donation from another person. The act involved donating a kidney. A generous act was performed by a woman in San Francisco. The act inspired others to donate kidneys. The act led to a chain of kidney transplants in San Francisco. A computer programmer received a kidney transplant. The computer programmer paid it forward. The computer programmer developed MatchGrid. That matched up donors with matching relatives. That used genetic data to match up donors with matching relatives."
    },
    {
        "fizz_score": 0.8575441837310791,
        "filtered_atomic_facts": "Woman's donation is matched. Woman's donation is matched with genetic profiles. Woman's donation is matched from a donor and recipient. That sends a chain reaction. The chain reaction is like dominoes falling. The chain of surgeries is to be wrapped up. The chain of surgeries is to be wrapped up on Friday. Doctors are performing an operation. The operation involves extracting kidneys. The operation involves inserting kidneys into recipients. The number of kidneys extracted and recipients is six. Long-chain transplanting is rare."
    },
    {
        "fizz_score": 0.47219958901405334,
        "filtered_atomic_facts": "A woman gave one of her kidneys to a stranger. Six people will receive kidney transplants. Five surgeons perform the surgeries. Other staff perform the surgeries. The chain of surgeries is to be wrapped up. The chain of surgeries is to be wrapped up on Friday. A computer programmer used his gift. The gift of the computer programmer was building a program. The program built by the computer programmer matches donor pairs. The program built by the computer programmer matches donor pairs quickly. A former patient posted a message. The message was posted on Facebook. The message was posted in her name."
    },
]

MENLI_SCORES = [
    {
        "entailment_menli": 0.0037422236055135727,
        "menli_cont": -0.045856982469558716,
        "menli_summ": -0.0037372747901827097
    },
    {
        "entailment_menli": 0.0017072205664590001,
        "menli_cont": -0.004461744800209999,
        "menli_summ": 0.0015694878529757261
    },
    {
        "entailment_menli": 0.0031177359633147717,
        "menli_cont": -0.0022557638585567474,
        "menli_summ": 0.005251282826066017
    },
    {
        "entailment_menli": 0.0014257538132369518,
        "menli_cont": -0.0018284340621903539,
        "menli_summ": 0.42649587988853455
    }
]

MOVERSCORE_SCORES = [
    {
        "moverscore": 0.5310592523579055
    },
    {
        "moverscore": 0.5640974366624864
    },
    {
        "moverscore": 0.5274455202368991
    },
    {
        "moverscore": 0.5487727863118251
    }
]

ROUGE_SCORES = [
    {
        "rouge1": 0.09999999999999999,
        "rouge2": 0.0,
        "rougeL": 0.049999999999999996,
        "rougeLsum": 0.09999999999999999
    },
    {
        "rouge1": 0.3513513513513513,
        "rouge2": 0.1388888888888889,
        "rougeL": 0.2702702702702703,
        "rougeLsum": 0.2702702702702703
    },
    {
        "rouge1": 0.14084507042253522,
        "rouge2": 0.0,
        "rougeL": 0.08450704225352113,
        "rougeLsum": 0.11267605633802817
    },
    {
        "rouge1": 0.29268292682926833,
        "rouge2": 0.05,
        "rougeL": 0.1951219512195122,
        "rougeLsum": 0.1951219512195122
    },
]

SIMCLS_SCORES = [
    {
        "simcls": 0.9994194507598877
    },
    {
        "simcls": 0.9990655779838562
    },
    {
        "simcls": 0.9988095760345459
    },
    {
        "simcls": 0.9991339445114136
    },
]

UNIEVAL_SCORES = [
    {
        "unieval_coherence": 0.49158343443988756,
        "unieval_consistency": 0.7056449673966408,
        "unieval_fluency": 0.7722202937082435,
        "unieval_relevance": 0.7727784207689579,
        "unieval_overall": 0.6855567790784325
    },
    {
        "unieval_coherence": 0.9110808660946638,
        "unieval_consistency": 0.7617173277491534,
        "unieval_fluency": 0.8207913769396674,
        "unieval_relevance": 0.9636276567323028,
        "unieval_overall": 0.8643043068789469
    },
    {
        "unieval_coherence": 0.9252296603167499,
        "unieval_consistency": 0.9309954427442108,
        "unieval_fluency": 0.881572622801874,
        "unieval_relevance": 0.8358401417745442,
        "unieval_overall": 0.8934094669093448
    },
    {
        "unieval_coherence": 0.5077862991819321,
        "unieval_consistency": 0.8110193336212775,
        "unieval_fluency": 0.927466363640644,
        "unieval_relevance": 0.7484631856268151,
        "unieval_overall": 0.7486837955176673
    }
]