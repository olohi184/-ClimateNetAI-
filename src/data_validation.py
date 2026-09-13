90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
        failures = int(converted.isna().sum() - df[col].isna().sum())
        pd.to_numeric(df["Month_Number"], errors="coerce").dropna().astype(int)
    )
    absent = sorted(EXPECTED_MONTHS - month_numbers)
    if absent:
        problems.append(f"Missing month numbers: {absent}")
        print(f"[FAIL] Missing month numbers: {absent}")
    else:
        print("[PASS] All 12 month numbers are represented.")

    counts = df.groupby(["Month_Number", "Month"]).size().sort_index()
    print("\nObservations by month:")
    for (month_no, month_name), count in counts.items():
        print(f"  {int(month_no):02d} {month_name}: {count}")

    # 6. September small-sample check
    september = df[df["Month"].astype(str).str.lower() == "september"]
    if len(september) <= 10:
        warnings.append(
            f"September has only {len(september)} observations; "
            "small-sample validation requires special care."
        )
        print(
            f"[WARN] September contains {len(september)} observations. "
            "Preserve the documented LOOCV/reconstructed-validation treatment."
        )

    # 7. June/July equivalence check
    june = df[df["Month"].astype(str).str.lower() == "june"][MODEL_COLUMNS].reset_index(drop=True)
    july = df[df["Month"].astype(str).str.lower() == "july"][MODEL_COLUMNS].reset_index(drop=True)

    if not june.empty and not july.empty and june.equals(july):
        warnings.append(
            "June and July contain identical Temperature, Pressure, "
            "Relative_Humidity and RSSI observations in the same order."
        )
        print(
            "[WARN] June and July modelling observations are identical. "
            "Do not interpret them as independent monthly evidence until "
            "the source data are verified."
        )
    else:
        print("[PASS] June and July modelling observations are not exactly identical.")

    # 8. Observed ranges (descriptive, not hard-coded validity limits)
    print("\nObserved modelling-variable ranges:")
    for col in MODEL_COLUMNS:
        numeric = pd.to_numeric(df[col], errors="coerce")
        print(f"  {col}: {numeric.min():.4f} to {numeric.max():.4f}")

    print("\n" + "-" * 64)
    if problems:
        print("Validation status: FAILED")
        for item in problems:
            print(f"  ERROR: {item}")
        return False

    print("Validation status: PASSED")
    if warnings:
        print(f"Warnings requiring research interpretation: {len(warnings)}")
        for item in warnings:
            print(f"  - {item}")
    else:
        print("No warnings detected.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a ClimateNetAI modelling dataset."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="modeling_dataset_v1.csv",
        help="Path to the modelling CSV (default: modeling_dataset_v1.csv)",
    )
    args = parser.parse_args()

    ok = validate_dataset(args.csv_path)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
