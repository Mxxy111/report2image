using System;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Windows.Forms;

internal static class NanoBananaPETCTLauncher
{
    private const string AppTitle = "NanoBanana PET-CT";
    private const string DefaultHost = "127.0.0.1";
    private const int DefaultPort = 8001;

    [STAThread]
    private static void Main()
    {
        Application.EnableVisualStyles();

        string projectRoot = FindProjectRoot();
        if (projectRoot.Length == 0)
        {
            ShowError("Cannot find webapp\\main.py. Put this launcher in the project folder.");
            return;
        }

        string host = Environment.GetEnvironmentVariable("PETCT_WEB_HOST");
        if (string.IsNullOrWhiteSpace(host))
        {
            host = DefaultHost;
        }

        int port = DefaultPort;
        string portValue = Environment.GetEnvironmentVariable("PETCT_WEB_PORT");
        int parsedPort;
        if (!string.IsNullOrWhiteSpace(portValue) && int.TryParse(portValue, out parsedPort))
        {
            port = parsedPort;
        }

        string baseUrl = string.Format("http://{0}:{1}", host, port);
        if (!IsHealthy(baseUrl))
        {
            string python = FindPython(projectRoot);
            if (python.Length == 0)
            {
                ShowError("Cannot find Python. Install Python or create .venv\\Scripts\\python.exe in the project folder.");
                return;
            }

            try
            {
                StartServer(projectRoot, python, host, port);
            }
            catch (Exception exc)
            {
                ShowError("Failed to start the local web service:\n" + exc.Message);
                return;
            }

            if (!WaitForHealthy(baseUrl, TimeSpan.FromSeconds(45)))
            {
                ShowError("The local web service did not become ready in time.\nPlease run scripts\\start_web.ps1 -Port 8001 to view logs.");
                return;
            }
        }

        OpenBrowser(baseUrl);
    }

    private static string FindProjectRoot()
    {
        string directory = AppDomain.CurrentDomain.BaseDirectory;
        for (int i = 0; i < 5 && !string.IsNullOrEmpty(directory); i++)
        {
            if (File.Exists(Path.Combine(directory, "webapp", "main.py")))
            {
                return directory;
            }
            directory = Directory.GetParent(directory) == null
                ? string.Empty
                : Directory.GetParent(directory).FullName;
        }
        return string.Empty;
    }

    private static string FindPython(string projectRoot)
    {
        string[] candidates = new[]
        {
            Path.Combine(projectRoot, ".venv", "Scripts", "python.exe"),
            Path.Combine(projectRoot, "venv", "Scripts", "python.exe"),
            "python.exe",
            "py.exe"
        };

        foreach (string candidate in candidates)
        {
            if (candidate.EndsWith(".exe", StringComparison.OrdinalIgnoreCase)
                && candidate.IndexOf(Path.DirectorySeparatorChar) >= 0
                && !File.Exists(candidate))
            {
                continue;
            }
            if (CanRunPython(candidate))
            {
                return candidate;
            }
        }
        return string.Empty;
    }

    private static bool CanRunPython(string python)
    {
        try
        {
            using (Process process = Process.Start(new ProcessStartInfo
            {
                FileName = python,
                Arguments = "--version",
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            }))
            {
                if (process == null)
                {
                    return false;
                }
                return process.WaitForExit(3000) && process.ExitCode == 0;
            }
        }
        catch
        {
            return false;
        }
    }

    private static void StartServer(string projectRoot, string python, string host, int port)
    {
        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = python,
            Arguments = "-m webapp.main",
            WorkingDirectory = projectRoot,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        startInfo.EnvironmentVariables["PETCT_WEB_HOST"] = host;
        startInfo.EnvironmentVariables["PETCT_WEB_PORT"] = port.ToString();
        Process.Start(startInfo);
    }

    private static bool WaitForHealthy(string baseUrl, TimeSpan timeout)
    {
        DateTime deadline = DateTime.UtcNow.Add(timeout);
        while (DateTime.UtcNow < deadline)
        {
            if (IsHealthy(baseUrl))
            {
                return true;
            }
            Thread.Sleep(1000);
        }
        return false;
    }

    private static bool IsHealthy(string baseUrl)
    {
        try
        {
            using (HttpClient client = new HttpClient())
            {
                client.Timeout = TimeSpan.FromSeconds(2);
                HttpResponseMessage response = client.GetAsync(baseUrl + "/api/health").GetAwaiter().GetResult();
                return response.IsSuccessStatusCode;
            }
        }
        catch
        {
            return false;
        }
    }

    private static void OpenBrowser(string baseUrl)
    {
        try
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = baseUrl,
                UseShellExecute = true
            });
        }
        catch (Exception exc)
        {
            ShowError("The service is running, but the browser could not be opened:\n" + exc.Message + "\n\nOpen manually: " + baseUrl);
        }
    }

    private static void ShowError(string message)
    {
        MessageBox.Show(message, AppTitle, MessageBoxButtons.OK, MessageBoxIcon.Error);
    }
}
