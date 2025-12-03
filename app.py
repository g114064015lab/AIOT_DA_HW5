import streamlit as st
from detector import detect_ai, _TRANSFORMERS_AVAILABLE


st.set_page_config(page_title='AI Text Detector', layout='wide')

st.title('AI Text Detector')

with st.sidebar:
    st.header('Options')
    use_gpt2 = st.checkbox('Use GPT-2 perplexity (may download model)', value=False)
    st.markdown('If GPT-2 is unavailable or you have limited RAM, uncheck this.')

input_text = st.text_area('Paste text to analyze', height=300)

if st.button('Detect'):
    if not input_text.strip():
        st.warning('Please paste some text to analyze.')
    else:
        with st.spinner('Analyzing...'):
            result = detect_ai(input_text, use_gpt2=use_gpt2)

        prob = result.get('probability', 0.0)
        label = result.get('label', 'Unclear')

        st.subheader(f'Result: {label} (probability {prob:.2f})')

        st.markdown('**Breakdown**')
        breakdown = result.get('breakdown', {})
        col1, col2 = st.columns(2)
        with col1:
            st.write('Features:')
            for k, v in breakdown.get('features', {}).items():
                st.write(f'- `{k}`: {v}')
        with col2:
            st.write('Parts:')
            for p in breakdown.get('parts', []):
                st.write(f'- `{p.get("name")}`: score={p.get("score"):.3f}, weight={p.get("weight"):.2f}')

        st.write('GPT-2 perplexity:', breakdown.get('gpt2_perplexity'))

        st.markdown('**Suggested prompt templates**')
        st.markdown('System prompt (Chinese):')
        st.code('''你是一個文本鑑定專家，判斷輸入文本是否由 AI 生成。返回 JSON: label, probability, explanation, evidence。''')

        st.markdown('User prompt example:')
        st.code('''請判斷以下文字是否由 AI 生成。只回傳 JSON。\n\n[TEXT]''')

        st.markdown('**Advice**')
        st.write('Use the GPT-2 perplexity option for a stronger signal if you have enough RAM and network; otherwise the heuristic score is still informative.')

st.markdown('---')
st.markdown('Created with `detector.py`. See `README.md` for full prompt templates and deployment notes.')
