from sentence_transformers import SentenceTransformer, util
import os

# Load both models for comparison
generic = SentenceTransformer('all-MiniLM-L6-v2')
# Path to the fine‑tuned model saved by the pipeline
fine_tuned_path = os.path.join(os.path.dirname(__file__), 'finetuned_event_model')
fine_tuned = SentenceTransformer(fine_tuned_path)

test_cases = [
    # (tweet, news_headline, should_be_similar)
    ('omg ground shaking in Istanbul 😱',
     '6.2 magnitude earthquake strikes Turkey', True),
    ('markets bleeding rn sell everything 📉',
     'Stock market sees sharp decline', True),
    ('new covid variant spreading fr',
     'WHO monitors new virus strain in Southeast Asia', True),
    ('skibidi toilet no cap lmao',
     '6.2 magnitude earthquake strikes Turkey', False),
    ('this cat is literally everything 😭',
     'WHO declares health emergency', False),
]

print('GENERIC   FINE‑TUNED   Expected   Status   Tweet')
print('=' * 80)

for tweet, news, should_be_similar in test_cases:
    e1_g = generic.encode(tweet)
    e2_g = generic.encode(news)
    score_g = util.cos_sim(e1_g, e2_g).item()
    
    e1_f = fine_tuned.encode(tweet)
    e2_f = fine_tuned.encode(news)
    score_f = util.cos_sim(e1_f, e2_f).item()
    
    expected = 'HIGH' if should_be_similar else 'LOW'
    status = '✅' if (
        (score_f > 0.7 and should_be_similar) or
        (score_f < 0.3 and not should_be_similar)
    ) else '❌'
    
    print(f'{score_g:.2f}      {score_f:.2f}      {expected:<8}   {status}   {tweet[:40]}')
