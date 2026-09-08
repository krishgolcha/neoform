import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const emptyList = { data: [] };

test.beforeEach(async ({ page }) => {
  await page.route("http://127.0.0.1:8799/api/v1/evolutions", async (route) => {
    await route.fulfill({ json: emptyList });
  });
  await page.route("http://127.0.0.1:8799/api/v1/doctor", async (route) => {
    await route.fulfill({ json: { ok: true, connected: true, provider: "tinker" } });
  });
});

test("renders the truthful empty lab and launch configuration", async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Evolution starts with a single checkpoint." }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Configure first evolution" }).click();
  await expect(page.getByRole("dialog", { name: "Design the search space" })).toBeVisible();
  await expect(page.getByText("Manual champion promotion")).toBeVisible();
});

test("has no serious accessibility violations", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name === "mobile", "One accessibility scan per markup variant");
  await page.goto("/");
  await expect(page.getByRole("main")).toBeVisible();
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations.filter((violation) => violation.impact === "serious")).toEqual([]);
});
