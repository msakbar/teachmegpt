import { Highlight, themes } from "prism-react-renderer";

const theme = {
  ...themes.github,
  plain: {
    ...themes.github.plain,
    backgroundColor: "var(--color-well)",
    color: "var(--color-ink-code)",
  },
};

export default function HighlightedCode({ code }) {
  return (
    <Highlight theme={theme} code={code.trim()} language="python">
      {({ style, tokens, getLineProps, getTokenProps }) => (
        <pre
          className="rounded-2xl p-5 overflow-x-auto text-[13px] leading-[1.7] font-mono"
          style={{ ...style, backgroundColor: "var(--color-well)" }}
        >
          {tokens.map((line, i) => (
            <div key={i} {...getLineProps({ line })}>
              <span className="inline-block w-8 text-right mr-4 text-ink-tertiary select-none text-[12px]">
                {i + 1}
              </span>
              {line.map((token, key) => (
                <span key={key} {...getTokenProps({ token })} />
              ))}
            </div>
          ))}
        </pre>
      )}
    </Highlight>
  );
}
