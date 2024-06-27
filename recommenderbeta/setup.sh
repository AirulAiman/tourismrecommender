mkdir -p ~/.streamlit/

echo "\
port = $PORT
enableCORS = false
headless = true

" > ~/.streamlit/config.toml