import { useState } from "react";
import "./App.css";

const percentages = [99, 90, 80, 70, 60, 50, 40, 30, 20, 10, 1];
const reasons = [
  "It got the syntax wrong",
  "It got the formatting wrong",
  "It is not idiomatic enough",
  "It didn't use a fact in the prompt",
  "It wasn't able to do reasoning",
  "It doesn't know a basic fact",
  "It doesn't know a domain specific fact",
  "I have no clue",
];

function randomBool() {
  return Math.random() < 0.5;
}

function getLeftToken(comparison, correctToTheLeft) {
  if (correctToTheLeft) {
    return comparison.correct_token_str;
  } else {
    return comparison.generated_token_str;
  }
}
function getRightToken(comparison, correctToTheLeft) {
  if (correctToTheLeft) {
    return comparison.generated_token_str;
  } else {
    return comparison.correct_token_str;
  }
}

function WhichOneApp(props) {
  const { initialComparison } = props;

  const [name, setName] = useState(localStorage.getItem("name") || "anon");

  const [comparison, setComparison] = useState(initialComparison);
  const [correctToTheLeft, setCorrectToTheLeft] = useState(randomBool());

  const [guess, setGuess] = useState(-1);
  const [hasGuessed, setHasGuessed] = useState(false);

  const [reason, setReason] = useState("");
  const [error, setError] = useState("");

  function getNewComparison() {
    fetch("/get_comparison")
      .then((response) => response.json())
      .then((comparison) => {
        setComparison(comparison);
      });
  }

  function reset() {
    setGuess(-1);
    setHasGuessed(false);
    getNewComparison();
    setCorrectToTheLeft(randomBool());
    setError("");
    setReason("");
  }
  function handleMainSubmit(e) {
    if (reason.length === 0) {
      setError("Please give a reason before submitting");
      return;
    }
    reset();
    fetch("/submit_whichone_guess", {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        username: name,
        guess: correctToTheLeft ? guess : 100 - guess,
        comparison_id: comparison.id,
        reason: reason,
      }),
    });
  }

  return (
    <div className="whichone-App">
      <h1>A new language modelling game!</h1>
      <p>
        Read instructions and notes{" "}
        <a href="https://docs.google.com/document/d/1BQF7EKs_kNssxZguyFK0X9pmtTYOF38qT-768vzNR1E/edit?usp=sharing">
          here
        </a>
        . This is a draft version of the app. What is needed: the data without
        duplicates, the google doc, better explanations. A better title ?
      </p>
      <div>
        Your name:{" "}
        <input
          value={name}
          onChange={(e) => {
            setName(e.target.value);
            localStorage.setItem("name", e.target.value);
          }}
        />
      </div>
      <p>
        Read the following prompt. Token A or token B is a token that appeared
        next index the original text. The other one was generated by a language
        model. How confident are you that <b>token A</b> is the one that
        appeared in the original text?
      </p>
      <p>(comparison number: {comparison.id})</p>
      <p className="prompt">{comparison.input_str}</p>
      <div className="token-list">
        <div className={correctToTheLeft && hasGuessed ? "token-correct" : ""}>
          <p>Token A</p>
          <p>
            <b>
              <pre>{getLeftToken(comparison, correctToTheLeft)}</pre>
            </b>
          </p>
        </div>
        <div className={!correctToTheLeft && hasGuessed ? "token-correct" : ""}>
          <p>Token B</p>
          <p>
            <b>
              <pre>{getRightToken(comparison, correctToTheLeft)}</pre>
            </b>
          </p>
        </div>
      </div>
      <div className="radios">
        <p>A is more likely to be correct</p>
        {percentages.map(function (percent, i) {
          return (
            <label key={`percentage_${i}`}>
              {percent}%<br />
              <input
                type="radio"
                id={`percentage_${i}`}
                value={percent}
                checked={guess === percent}
                onChange={function () {
                  if (!hasGuessed) {
                    setGuess(percent);
                    setError("");
                  }
                }}
              />
            </label>
          );
        })}
        <p>B is more likely to be correct</p>
      </div>
      <div className="submit-button">
        <button
          onClick={function () {
            if (guess === -1) {
              setError("Please give your confidence level.");
              return;
            }
            setHasGuessed(true);
          }}
        >
          Submit guess
        </button>
        {/* <span style={{ width: "2em" }} />
        <button onClick={reset}>Skip this completion</button> */}
      </div>
      {hasGuessed && (
        <div>
          <p>
            The correct token was token is token {correctToTheLeft ? "A" : "B"}
          </p>
          <p>
            Why did the model fail to generate the correct token ? Give the
            description that best matches the difference between the model
            generated completion and the true one.
          </p>
          <div className="reasons-radios">
            {reasons.map(function (r, i) {
              return (
                <label key={`reason_${i}`}>
                  <input
                    type="radio"
                    id={`reason_${i}`}
                    value={r}
                    checked={reason === r}
                    onClick={function () {
                      setReason(r);
                      setError("");
                    }}
                  />
                  {r}
                </label>
              );
            })}
            <label>
              <span style={{ marginRight: "6px" }}>Other</span>
              <input
                type="text"
                value={reason}
                onChange={function (e) {
                  setReason(e.target.value);
                  setError("");
                }}
                style={{ width: "18em" }}
              ></input>
            </label>
          </div>
          <div className="submit-button">
            <button onClick={handleMainSubmit}>
              Submit and get new comparison
            </button>
          </div>
        </div>
      )}
      <p className="error">{error}</p>
      <br />
    </div>
  );
}

export default WhichOneApp;
