import { SECTIONS } from "../../data/sections";
import { SectionBlock } from "../content/SectionBlock";

export function SectionContainer({ setSectionRef }) {
  return (
    <main className="max-w-[640px] md:max-w-[1140px] mx-auto px-7 pt-12 pb-20">
      {SECTIONS.map((section, i) => (
        <div key={section.id}>
          {i > 0 ? (
            <div className="border-t border-[rgba(0,0,0,0.04)]" />
          ) : null}
          <SectionBlock
            section={section}
            sectionRef={setSectionRef(i)}
          />
        </div>
      ))}
    </main>
  );
}
