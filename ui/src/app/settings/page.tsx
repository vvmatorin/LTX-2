"use client";

import { useState } from "react";
import { useSettings } from "@/hooks/useSettings";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { PageHeader } from "@/components/PageHeader";
import { Save, Loader2 } from "lucide-react";
import type { AppSettings } from "@/lib/types";

function buildLocalSettings(api: AppSettings): AppSettings {
  return {
    modelPath: api.modelPath || "",
    textEncoderPath: api.textEncoderPath || "",
    outputDir: api.outputDir || "",
    datasetDir: api.datasetDir || "",
    scriptsDir: api.scriptsDir || "",
  };
}

export default function SettingsPage() {
  const { settings: apiSettings, saveSettings, isSaving } = useSettings();

  if (!apiSettings) {
    return (
      <div className="flex items-center justify-center p-12">
        <Loader2 className="text-muted-foreground h-6 w-6 animate-spin" />
      </div>
    );
  }

  return <SettingsForm initial={apiSettings} saveSettings={saveSettings} isSaving={isSaving} />;
}

function SettingsForm({
  initial,
  saveSettings,
  isSaving,
}: {
  initial: AppSettings;
  saveSettings: (s: Partial<AppSettings>) => Promise<unknown>;
  isSaving: boolean;
}) {
  const [local, setLocal] = useState<AppSettings>(() => buildLocalSettings(initial));
  const [saved, setSaved] = useState(false);

  const handleSave = async () => {
    try {
      await saveSettings(local);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch {
      // react-query surfaces the error via isError
    }
  };

  const update = (key: keyof AppSettings, value: string) => {
    setLocal((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-3 md:p-4">
      <PageHeader title="Settings" subtitle="Global defaults for model paths and directories" />

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Model Paths</CardTitle>
          <CardDescription>
            Default model and text encoder paths used across all jobs
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="modelPath">LTX Checkpoint Path</Label>
            <Input
              id="modelPath"
              value={local.modelPath}
              onChange={(e) => update("modelPath", e.target.value)}
              placeholder="/path/to/ltx-2.3-22b-dev.safetensors"
              className="font-mono text-xs"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="textEncoderPath">Text Encoder Path</Label>
            <Input
              id="textEncoderPath"
              value={local.textEncoderPath}
              onChange={(e) => update("textEncoderPath", e.target.value)}
              placeholder="/path/to/google/gemma-3-12b-it"
              className="font-mono text-xs"
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Directories</CardTitle>
          <CardDescription>Output and scripts paths</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="outputDir">Training Output Directory</Label>
            <Input
              id="outputDir"
              value={local.outputDir}
              onChange={(e) => update("outputDir", e.target.value)}
              placeholder="/path/to/training/outputs"
              className="font-mono text-xs"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="datasetDir">Dataset Directory</Label>
            <Input
              id="datasetDir"
              value={local.datasetDir}
              onChange={(e) => update("datasetDir", e.target.value)}
              placeholder="/path/to/preprocessed/datasets"
              className="font-mono text-xs"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="scriptsDir">Scripts Directory</Label>
            <Input
              id="scriptsDir"
              value={local.scriptsDir}
              onChange={(e) => update("scriptsDir", e.target.value)}
              placeholder="/path/to/LTX-2/packages/ltx-trainer/scripts"
              className="font-mono text-xs"
            />
          </div>
        </CardContent>
      </Card>

      <div className="flex justify-end">
        <Button onClick={handleSave} disabled={isSaving}>
          <Save className="mr-1.5 h-4 w-4" />
          {saved ? "Saved!" : isSaving ? "Saving..." : "Save Settings"}
        </Button>
      </div>
    </div>
  );
}
