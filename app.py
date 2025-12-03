import streamlit as st
from detector import detect_ai, _TRANSFORMERS_AVAILABLE


st.set_page_config(page_title='AI Text Detector', layout='wide')

SAMPLE_TEXTS = {
    'Manual input': '',
    'Academic summary': (
        'Large language models are increasingly woven into daily productivity tools, '
        'raising new expectations for educators to verify authorship and originality.'
    ),
    'Conversational reply': (
        'Hey! I skimmed your notes and the project goals look solid - let me know if you want '
        'me to help polish the final conclusion later tonight.'
    ),
    'Creative paragraph': (
        'Sunlight spilled across the workshop bench, catching dust motes in the air as the '
        'prototype drone hummed awake for its first autonomous test flight.'
    ),
}


if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ''
if 'use_gpt2_option' not in st.session_state:
    st.session_state['use_gpt2_option'] = False
if 'last_result' not in st.session_state:
    st.session_state['last_result'] = None

st.title('AI Text Detector')

with st.sidebar:
    st.title('Detector Toolkit')
    st.caption('Tune options, explore samples, and review the latest verdict.')
    st.divider()

    st.subheader('Run options')
    st.checkbox('Use GPT-2 perplexity (may download model)', key='use_gpt2_option')
    use_gpt2 = st.session_state.get('use_gpt2_option', False)
    if use_gpt2 and not _TRANSFORMERS_AVAILABLE:
        st.warning('transformers/torch not installed. Falling back to heuristics.')
    elif use_gpt2:
        st.success('GPT-2 enabled for deeper analysis.')
    else:
        st.info('Heuristic-only mode keeps things lightweight.')

    st.divider()
    st.subheader('Quick samples')
    sample_choice = st.selectbox('Load preset text', list(SAMPLE_TEXTS.keys()), index=0)
    load_col, clear_col = st.columns(2)
    with load_col:
        if st.button('Apply sample', use_container_width=True):
            st.session_state['input_text'] = SAMPLE_TEXTS[sample_choice]
    with clear_col:
        if st.button('Clear text', use_container_width=True):
            st.session_state['input_text'] = ''
            st.session_state['last_result'] = None

    st.caption('Samples are editable once loaded into the main editor.')

    with st.expander('Scoring cheat sheet'):
        st.write('- 60% - GPT-2 perplexity (if available)')
        st.write('- 15% - Repetition of trigram phrases')
        st.write('- 10% - Ratio of short connective words')
        st.write('- 15% - Punctuation density and variety')

    last = st.session_state.get('last_result')
    if last:
        st.divider()
        st.subheader('Last verdict')
        probability = last.get('probability', 0.0)
        label = last.get('label', 'Unclear')
        st.metric('AI likelihood', f'{probability * 100:.1f}%')
        st.caption(f'Label: **{label}**')

input_text = st.text_area('Paste text to analyze', height=300, key='input_text')

if st.button('Detect', use_container_width=True):
    if not input_text.strip():
        st.warning('Please paste some text to analyze.')
    else:
        with st.spinner('Analyzing...'):
            result = detect_ai(input_text, use_gpt2=st.session_state.get('use_gpt2_option', False))

        prob = result.get('probability', 0.0)
        label = result.get('label', 'Unclear')
        st.session_state['last_result'] = result

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
        st.code('''ä½ æ˜¯ä¸€?‹æ??¬é?å®šå?å®¶ï??¤æ–·è¼¸å…¥?‡æœ¬?¯å¦??AI ?Ÿæ??‚è???JSON: label, probability, explanation, evidence??''')

        st.markdown('User prompt example:')
        st.code('''è«‹åˆ¤?·ä»¥ä¸‹æ?å­—æ˜¯?¦ç”± AI ?Ÿæ??‚åª?žå‚³ JSON?‚

[TEXT]''')

        st.markdown('**Advice**')
        st.write('Use the GPT-2 perplexity option for a stronger signal if you have enough RAM and network; otherwise the heuristic score is still informative.')

st.markdown('---')
st.markdown('Created with `detector.py`. See `README.md` for full prompt templates and deployment notes.')
