use dotenv::dotenv;
// use http_req::{request::Method, request::Request, uri::Uri};
use lambda_flows::{request_received, send_response};
use reqwest::{header, Client, ClientBuilder, Error, Response};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use tokio;
use web_scraper_flows::get_page_text;

#[no_mangle]
#[tokio::main(flavor = "current_thread")]
pub async fn run() {
    request_received(handler).await;
}

async fn handler(_qry: HashMap<String, Value>, _body: Vec<u8>) {
    let url = _qry.get("url").unwrap().as_str().unwrap();
    match get_page_text(&url).await {
        Ok(text) => {
            let sys_prompt = "You're an AI assistant";
            let u_prompt = format!("summarize this: {}", text);
            let res = custom_gpt(sys_prompt, &u_prompt, 128)
                .await
                .unwrap_or("no summary".to_string());

            send_response(
                200,
                vec![(String::from("content-type"), String::from("text/html"))],
                res.as_bytes().to_vec(),
            )
        }

        Err(_e) => send_response(
            200,
            vec![(String::from("content-type"), String::from("text/html"))],
            _e.as_bytes().to_vec(),
        ),
    };
}

pub async fn custom_gpt(sys_prompt: &str, u_prompt: &str, m_token: u16) -> Option<String> {
    let system_prompt = serde_json::json!(
        {"role": "system", "content": sys_prompt}
    );
    let user_prompt = serde_json::json!(
        {"role": "user", "content": u_prompt}
    );

    match chat(vec![system_prompt, user_prompt], m_token).await {
        Ok((res, _count)) => Some(res),
        Err(_) => None,
    }
}
pub async fn chat(message_obj: Vec<Value>, m_token: u16) -> Result<(String, u32), anyhow::Error> {
    dotenv().ok();
    let api_token = env::var("OPENAI_API_TOKEN").unwrap();

    let mut headers = header::HeaderMap::new();
    let bearer_token = format!("Bearer {}", api_token);
    let auth_value = header::HeaderValue::from_str(&bearer_token)?;
    headers.insert(header::AUTHORIZATION, auth_value);

    let params = serde_json::json!({
        "model": "gpt-3.5-turbo-16k",
        "messages": message_obj,
        "temperature": 0.7,
        "top_p": 1,
        "n": 1,
        "stream": false,
        "max_tokens": m_token,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "stop": "\n"
    });

    let client = Client::builder().default_headers(headers).build()?;

    let uri = "https://api.openai.com/v1/chat/completions";

    let res = client.post(uri).json(&params).send().await?;

    let chat_response: ChatResponse = res.json().await?;
    let token_count = chat_response.usage.total_tokens;

    Ok((
        chat_response.choices[0].message.content.to_string(),
        token_count,
    ))
}

#[derive(Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}
